"""CPU invariants: exact damage scope, trial sampling, and hook single-fire."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

f=importlib.import_module('fault_injection.2026-10-17_sglang_fault_models')
c=importlib.import_module('experiments.2026-10-17_sglang_fault_model_campaign')


class FaultModelsTests(unittest.TestCase):
    def test_repeated_compute_balances_each_input_and_changes_seed(self):
        inventory=[dict(module='model.layers.0.mlp.down_proj',numel=64)]
        seeds=set()
        for sid in range(100):
            plan=[c.choose_command('compute',sid,inventory,20261019,40,trial_id=t) for t in range(20)]
            self.assertEqual(plan,[c.choose_command('compute',sid,inventory,20261019,40,trial_id=t) for t in range(20)])
            for phase in ('prefill','target_verify'):
                for bits in (1,2):
                    self.assertEqual(sum(x['phase']==phase and len(x['bit_positions'])==bits for x in plan),5)
            for command in plan:
                self.assertNotIn(command['seed'],seeds)
                seeds.add(command['seed'])
        plan=[c.choose_command('compute',0,inventory,20261019,40,trial_id=t) for t in range(20)]
        self.assertTrue(any(x['occurrence']>8 for x in plan if x['phase']=='target_verify'))
        self.assertNotEqual(plan[0],c.choose_command('compute',0,inventory,20261020,40,trial_id=0))

    def test_fused_output_component_uses_actual_partitions(self):
        module=NS(output_partition_sizes=[4,2,2])
        self.assertEqual([f.projection_component('qkv_proj',module,col,8) for col in (0,3,4,5,6,7)],
                         ['q','q','k','k','v','v'])
        module=NS(output_partition_sizes=[6,6])
        self.assertEqual(f.projection_component('gate_up_proj',module,7,12),'up')
        self.assertEqual(f.projection_component('gate_up_proj',module,7,16),'unknown_partition')

    def test_repeated_requests_have_distinct_artifacts_and_one_event(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            server=NS(manifest={'context_path':str(root/'context.json'),'arm_path':str(root/'arm.json')},process=object())
            args=NS(max_new_tokens=64)
            sample={'sample_id':0,'question':'q','gold_number':'3','prompt_sha256':'h',
                    'request':{'sampling_params':{}}}
            def guarded(_server,request,trace,event_path,*unused):
                command=json.loads((root/'arm.json').read_text())
                event={**command,'evidence':{'cells':[{'row':0,'column':0,'before':'3f80','after':'3f81','xor':'0001'}]}}
                c.save(event_path,event)
                events=[{'kind':'fault_context'}, {'kind':'verify_block','committed_token_ids':[2]}]
                trace.write_text(''.join(json.dumps(e)+'\n' for e in events))
                return {'text':'#### 3','output_ids':[1,2],'meta_info':{'cached_tokens':0,'spec_verify_ct':1}},None
            records=[]
            with patch.object(c.prior,'guarded_request',side_effect=guarded):
                for trial_id in range(2):
                    command={'kind':'compute','trial_id':trial_id,'seed':trial_id}
                    record=c.run_request(server,root/f'trial_{trial_id}',sample,command,args,f'compute_trial_{trial_id}')
                    self.assertTrue(record['checks_pass'])
                    self.assertEqual(record['trial_id'],trial_id)
                    records.append(record)
            self.assertNotEqual(records[0]['request']['rid'],records[1]['request']['rid'])
            self.assertNotEqual(records[0]['trace_path'],records[1]['trace_path'])
            self.assertFalse((root/'context.json').exists())
            self.assertFalse((root/'arm.json').exists())

    def test_position_analysis_uses_baseline_correct_denominator(self):
        records=[]
        for sid,baseline_correct in ((0,True),(1,False)):
            for trial_id in range(2):
                command={'trial_id':trial_id,'seed':10+sid*2+trial_id,'module':'model.layers.2.mlp.down_proj',
                         'phase':'prefill','occurrence':1,'bit_positions':[6],'event_path':'event.json'}
                changed=trial_id==1
                record={'sample_id':sid,'fault_command':command,'fault_event':{'evidence':{'cells':[
                        {'row':0,'column':1,'before':'3f80','after':'3fc0'}]}},'response':{},
                        'checks_pass':True,'answer_correct':baseline_correct and not changed,'trace_path':'trace.jsonl',
                        'comparison':{'output_ids_equal':not changed,'trial_accept_rate':.1,'baseline_accept_rate':.1,
                                      'trial_output_tokens':10,'baseline_correct_to_wrong':baseline_correct and changed}}
                records.append(c.compute_trial_row(record,{'answer_correct':baseline_correct}))
        result=c.position_analysis(records)
        self.assertEqual(result['overall']['baseline_correct_completed_trials'],2)
        self.assertEqual(result['overall']['correct_to_wrong'],1)
        self.assertEqual(result['overall']['harmful_rate_given_baseline_correct'],.5)
        self.assertEqual(result['inputs_with_mixed_harmful_outcomes'],1)
        self.assertEqual(result['overall']['finite_injection_harmful'],1)
        json.dumps(result,allow_nan=False)

    def test_single_and_double_bits_change_only_selected_cell(self):
        for positions in ([0],[7,14]):
            x=torch.arange(20,dtype=torch.bfloat16).reshape(4,5)
            original=x.view(torch.int16).clone()
            e=f.damage(x,dict(seed=10,pattern='bit',row=2,column=3,bit_positions=positions))
            expected=original.clone()
            expected[2,3]^=sum(1<<b for b in positions)
            self.assertTrue(torch.equal(x.view(torch.int16),expected))
            self.assertEqual(e['changed_elements'],1)

    def test_region_only_changes_requested_square(self):
        x=torch.ones((8,8),dtype=torch.bfloat16)
        before=x.clone()
        e=f.damage(x,dict(seed=2,pattern='region_zero',side=2,row=1,column=4))
        before[1:3,4:6]=0
        self.assertTrue(torch.equal(x,before))
        self.assertEqual(e['changed_elements'],4)
        with self.assertRaises(ValueError):
            f.damage(x[:1],dict(seed=1,pattern='region_random',side=4))

    def test_word_random_reproduces_raw_bits_and_does_not_leave_identical_cell(self):
        a=torch.ones((4,4),dtype=torch.bfloat16);b=a.clone()
        command=dict(seed=5,pattern='word_random')
        ea=f.damage(a,command);eb=f.damage(b,command)
        self.assertEqual(ea,eb)
        self.assertEqual(ea['changed_elements'],1)
        self.assertTrue(torch.equal(a.view(torch.int16),b.view(torch.int16)))

    def test_refuse_duplicate_bits_wrong_dtype_or_out_of_range(self):
        for positions in ([1,1],[16],[]):
            with self.assertRaises(ValueError):
                f.damage(torch.ones((2,2),dtype=torch.bfloat16),dict(seed=1,pattern='bit',bit_positions=positions))
        with self.assertRaises(TypeError):
            f.damage(torch.ones((2,2)),dict(seed=1,pattern='bit',bit_positions=[0]))

    def test_schedule_is_reproducible_and_balances_kv_subgroups(self):
        inventory=[dict(module='model.layers.0.mlp.down_proj',numel=64)]
        a=[c.choose_command('kv',i,inventory,12,5) for i in range(100)]
        b=[c.choose_command('kv',i,inventory,12,5) for i in range(100)]
        self.assertEqual(a,b)
        self.assertEqual(len({e['seed'] for e in a}),100)
        self.assertTrue(all(1<=e['occurrence']<=5 for e in a))
        for region in ('history','candidate'):
            for component in ('key','value'):
                subset=[e for e in a if e['region']==region and e['component']==component]
                self.assertEqual(len(subset),25)
                self.assertEqual({len(e['bit_positions']) for e in subset},{1,2})

    def test_bit_count_and_phase_are_not_confounded(self):
        inventory=[dict(module='model.layers.0.mlp.down_proj',numel=64)]
        cells=[c.choose_command('compute',i,inventory,12,5) for i in range(100)]
        for phase in ('prefill','target_verify'):
            for count in (1,2):
                self.assertEqual(sum(e['phase']==phase and len(e['bit_positions'])==count for e in cells),25)

    def test_result_hook_scope_single_fire_tuple_and_private_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);arm=root/'2026-10-17_arm.json';arm.write_text('{}')
            s=f.Session(dict(arm_path=str(arm)))
            s.context=dict(sample_id=0,rid='trial',trace_path=str(root/'2026-10-17_trace.jsonl'))
            s.active=dict(batch=NS(forward_mode=NS(name='TARGET_VERIFY')),index=2)
            s.command=dict(kind='compute',phase='target_verify',occurrence=2,module='linear',
                           seed=1,pattern='bit',bit_positions=[7],rid='trial',
                           event_path=str(root/'2026-10-17_event.json'))
            hook=s.result_hook('linear')
            x=torch.ones((2,3),dtype=torch.bfloat16);original=x.clone()
            self.assertIsNone(hook(None,None,(x,None)))
            output=hook(None,None,(x,None))
            self.assertIsInstance(output,tuple)
            self.assertTrue(torch.equal(x,original))
            self.assertIsNone(output[1])
            self.assertIsNone(hook(None,None,(x,None)))
            self.assertFalse(arm.exists())
            self.assertEqual(len((root/'2026-10-17_trace.jsonl').read_text().splitlines()),1)

    def test_weight_restore_changes_no_other_elements(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,2,bias=False,dtype=torch.bfloat16))
        before=model[0].weight.view(torch.int16).clone()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            f.weights._inject(model,dict(fault_id='w',module_name='0',flat_index=2,
                 bit_positions=[0,14],event_path=str(p/'2026-10-17_inject.json')))
            f.weights._restore(model,dict(fault_id='w',event_path=str(p/'2026-10-17_restore.json')))
            self.assertTrue(torch.equal(model[0].weight.view(torch.int16),before))
            event=json.loads((p/'2026-10-17_restore.json').read_text())
            self.assertEqual(event['status'],'restored')


if __name__=='__main__':
    unittest.main()

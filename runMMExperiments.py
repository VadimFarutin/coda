# Run on different "full" depths
# Re-run roadmap experiments 
# Map all scRNA stuff

import os
import time
import copy
import tempfile
import json
from subprocess import call
import wandb
from diConstants import (HG19_ALL_CHROMS, MM9_ALL_CHROMS,
    HG19_TRAIN_CHROMS, MM9_TRAIN_CHROMS,
    VALID_CHROMS, TEST_CHROMS) 
from modelPresetParams import MODEL_PRESET_PARAMS

import models
import modelTemplates


def run_model(model_params, evaluate, evaluate_genome_only):
    print("pre init")
    start = time.time()
    m = models.SeqModel.instantiate_model(model_params)
    print("init done")
    t1 = time.time()
    m.compile_and_train_model()
    t2 = time.time()
    results = None
    if evaluate:
        results = m.evaluate_model(evaluate_genome_only)
    t3 = time.time()
    print(f"Init: {t1 - start}, Train: {t2 - t1}, Evaluate: {t3 - t2}")

    return results


MM_MARKS = ['H3K4ME3']


def test_MOUSE():

    for test_cell_line in ['MOUSE']:
        for subsample_target_string in ['0.5e6']:
            for predict_binary_output in [False]:
                for output_mark in MM_MARKS:                            
                    model_type = 'cnn-encoder-decoder'
                    wandb_log = True
                    evaluate = True
                    evaluate_genome_only = True
                    
                    preset_params = MODEL_PRESET_PARAMS[model_type]
                    loss = preset_params['compile_params']['class_loss'] \
                           if predict_binary_output \
                           else preset_params['compile_params']['regression_loss']

                    model_params = modelTemplates.make_model_params(
                        model_library=preset_params['model_library'],
                        model_class=preset_params['model_class'],
                        model_type=preset_params['model_type'],
                        model_specific_params=preset_params['model_specific_params'],
                        compile_params={
                            'loss': loss,
                            'optimizer': preset_params['compile_params']['optimizer'],
                            'lr': preset_params['compile_params']['lr']
                        },
                        dataset_params={
                            'train_dataset_name': 'MOUSE_3marks_all',
                            'test_dataset_name': '%s_3marks_all' % test_cell_line,
                            'num_train_examples': 100000,
                            'seq_length': 1001,
                            'peak_fraction': 0.5,
                            'train_X_subsample_target_string': subsample_target_string,
                            'num_bins_to_test': None,
                            'train_chroms': MM9_TRAIN_CHROMS,
                            'test_chroms': VALID_CHROMS,
                            'only_chr1': False,
                            'wout_peaks': True
                        },
                        output_marks=[output_mark],
                        #input_marks=[output_mark, 'INPUT'],
                        train_params={
                            'nb_epoch': 0,
                            'batch_size': 100,
                            'validation_split': 0.2,
                            'wandb_log': wandb_log,
                            'eval_batch_size': 1000,
                        },
                        predict_binary_output=predict_binary_output,
                        zero_out_non_bins=True,
                        generate_bigWig=False,
                        #pretrained_model_path=None)
                        pretrained_model_path='./models/weights/cnn-encoder-decoder-20200506-091853444412-weights.pt')

                    if wandb_log:
                        group = "peaks" if predict_binary_output else "signal"
                        name = model_type + "_" + output_mark
                        # Initilize a new wandb run
                        wandb.init(entity="vadim-farutin", project="coda-mm", name=name,
                                   reinit=True,
                                   config=model_params, group=group, tags=[output_mark, model_type])
                        # wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

                    results = run_model(model_params, evaluate, evaluate_genome_only)

                    if wandb_log:
                        if predict_binary_output:
                            if not evaluate_genome_only:
                                wandb.run.summary['train_samples_dn_AUC']              = results['train_samples']['dn']['AUC']
                                wandb.run.summary['train_samples_dn_AUPRC']            = results['train_samples']['dn']['AUPRC']
                                wandb.run.summary['train_samples_dn_Y_pos_frac']       = results['train_samples']['dn']['Y_pos_frac']
                                wandb.run.summary['train_samples_dn_precision_curves'] = results['train_samples']['dn']['precision_curves']
                                wandb.run.summary['train_samples_dn_recall_curves']    = results['train_samples']['dn']['recall_curves']

                                wandb.run.summary['test_samples_dn_AUC']              = results['test_results'][0]['samples']['dn']['AUC']
                                wandb.run.summary['test_samples_dn_AUPRC']            = results['test_results'][0]['samples']['dn']['AUPRC']
                                wandb.run.summary['test_samples_dn_Y_pos_frac']       = results['test_results'][0]['samples']['dn']['Y_pos_frac']
                                wandb.run.summary['test_samples_dn_precision_curves'] = results['test_results'][0]['samples']['dn']['precision_curves']
                                wandb.run.summary['test_samples_dn_recall_curves']    = results['test_results'][0]['samples']['dn']['recall_curves']

                            wandb.run.summary['test_genome_dn_AUC']              = results['test_results'][0]['genome']['dn_all']['chr1']['AUC']
                            wandb.run.summary['test_genome_dn_AUPRC']            = results['test_results'][0]['genome']['dn_all']['chr1']['AUPRC']
                            wandb.run.summary['test_genome_dn_Y_pos_frac']       = results['test_results'][0]['genome']['dn_all']['chr1']['Y_pos_frac']
                            wandb.run.summary['test_genome_dn_precision_curves'] = results['test_results'][0]['genome']['dn_all']['chr1']['precision_curves']
                            wandb.run.summary['test_genome_dn_recall_curves']    = results['test_results'][0]['genome']['dn_all']['chr1']['recall_curves']

                        else:
                            if not evaluate_genome_only:
                                wandb.run.summary['train_samples_dn_MSE']      = results['train_samples']['dn']['MSE']
                                wandb.run.summary['train_samples_dn_true_var'] = results['train_samples']['dn']['true_var']
                                wandb.run.summary['train_samples_dn_pearsonR'] = results['train_samples']['dn']['pearsonR']
                                wandb.run.summary['train_samples_dn_SNR']      = results['train_samples']['dn']['SNR']

                                wandb.run.summary['test_samples_dn_MSE']      = results['test_results'][0]['samples']['dn']['MSE']
                                wandb.run.summary['test_samples_dn_true_var'] = results['test_results'][0]['samples']['dn']['true_var']
                                wandb.run.summary['test_samples_dn_pearsonR'] = results['test_results'][0]['samples']['dn']['pearsonR']
                                wandb.run.summary['test_samples_dn_SNR']      = results['test_results'][0]['samples']['dn']['SNR']

                            wandb.run.summary['test_genome_dn_all_MSE']      = results['test_results'][0]['genome']['dn_all']['chr1']['MSE']
                            wandb.run.summary['test_genome_dn_all_true_var'] = results['test_results'][0]['genome']['dn_all']['chr1']['true_var']
                            wandb.run.summary['test_genome_dn_all_pearsonR'] = results['test_results'][0]['genome']['dn_all']['chr1']['pearsonR']
                            wandb.run.summary['test_genome_dn_all_SNR']      = results['test_results'][0]['genome']['dn_all']['chr1']['SNR']

                            wandb.run.summary['test_genome_dn_peaks_MSE']      = results['test_results'][0]['genome']['dn_peaks']['chr1']['MSE']
                            wandb.run.summary['test_genome_dn_peaks_true_var'] = results['test_results'][0]['genome']['dn_peaks']['chr1']['true_var']
                            wandb.run.summary['test_genome_dn_peaks_pearsonR'] = results['test_results'][0]['genome']['dn_peaks']['chr1']['pearsonR']
                            wandb.run.summary['test_genome_dn_peaks_SNR']      = results['test_results'][0]['genome']['dn_peaks']['chr1']['SNR']

                        wandb.join()
                        # wandb.uninit()


if __name__ == '__main__':
    test_MOUSE()
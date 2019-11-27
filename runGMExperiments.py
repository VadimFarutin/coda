# Run on different "full" depths
# Re-run roadmap experiments 
# Map all scRNA stuff

import os
import copy
import tempfile
import json
from subprocess import call
import wandb
from diConstants import (HG19_ALL_CHROMS, MM9_ALL_CHROMS,
    HG19_TRAIN_CHROMS, MM9_TRAIN_CHROMS,
    VALID_CHROMS, TEST_CHROMS) 

import models
import modelTemplates

def run_model(model_params):
    m = models.SeqModel.instantiate_model(model_params)
    m.compile_and_train_model()
    results = m.evaluate_model()
    return results

GM_MARKS = ['H3K27AC', 'H3K4ME1', 'H3K4ME3', 'H3K27ME3', 'H3K36ME3']


def test_GM18526():

    for test_cell_line in ['GM18526']:
        for subsample_target_string in ['0.5e6']:
            for predict_binary_output in [True, False]:    
                for output_mark in GM_MARKS:                            

                    model_params = modelTemplates.make_model_params(
                        model_library='keras',
                        model_class='SeqToPoint',
                        model_type='cnn',
                        model_specific_params={
                            'num_filters': 6,
                            'filter_length': 51
                        },
                        compile_params={            
                            'optimizer': 'adagrad'
                        },
                        dataset_params={
                            'train_dataset_name': 'GM12878_5+1marks-K4me3_all',
                            'test_dataset_name': '%s_5+1marks-K4me3_all' % test_cell_line, 
                            'num_train_examples': 100000,
                            'seq_length': 1001,
                            'peak_fraction': 0.5,                            
                            'train_X_subsample_target_string': subsample_target_string,
                            'num_bins_to_test': None,
                            'train_chroms': HG19_ALL_CHROMS,
                            'test_chroms': HG19_ALL_CHROMS,
                            'only_chr1': True
                        },
                        output_marks=[output_mark],
                        train_params={
                            'nb_epoch': 30,
                            'batch_size': 100
                        },
                        predict_binary_output=predict_binary_output,
                        zero_out_non_bins=True,
                        generate_bigWig=True)

                    group = "peaks" if predict_binary_output else "signal"
                    # Initilize a new wandb run
                    wandb.init(entity="vadim-farutin", project="coda",
                               config=model_params, reinit=True, group=group)
                    # wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

                    results = run_model(model_params)

                    if predict_binary_output:
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
                        wandb.run.summary['train_samples_dn_MSE']      = results['train_samples']['dn']['MSE']
                        wandb.run.summary['train_samples_dn_true_var'] = results['train_samples']['dn']['true_var']
                        wandb.run.summary['train_samples_dn_pearsonR'] = results['train_samples']['dn']['pearsonR']

                        wandb.run.summary['test_samples_dn_MSE']      = results['test_results'][0]['samples']['dn']['MSE']
                        wandb.run.summary['test_samples_dn_true_var'] = results['test_results'][0]['samples']['dn']['true_var']
                        wandb.run.summary['test_samples_dn_pearsonR'] = results['test_results'][0]['samples']['dn']['pearsonR']

                        wandb.run.summary['test_genome_dn_all_MSE']      = results['test_results'][0]['genome']['dn_all']['chr1']['MSE']
                        wandb.run.summary['test_genome_dn_all_true_var'] = results['test_results'][0]['genome']['dn_all']['chr1']['true_var']
                        wandb.run.summary['test_genome_dn_all_pearsonR'] = results['test_results'][0]['genome']['dn_all']['chr1']['pearsonR']

                        wandb.run.summary['test_genome_dn_peaks_MSE']      = results['test_results'][0]['genome']['dn_peaks']['chr1']['MSE']
                        wandb.run.summary['test_genome_dn_peaks_true_var'] = results['test_results'][0]['genome']['dn_peaks']['chr1']['true_var']
                        wandb.run.summary['test_genome_dn_peaks_pearsonR'] = results['test_results'][0]['genome']['dn_peaks']['chr1']['pearsonR']

                    # wandb.join()
                    wandb.uninit()


if __name__ == '__main__':
    test_GM18526()

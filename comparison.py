import pandas as pd
import matplotlib.pyplot as plt
from dataset import Dataset


class Comparison:
    def __init__(self, dataset=Dataset.movielens_1m):
        self.title_text = '1M'
        self.__dataset = dataset
        self.__selected_ds, _ = self.__dataset.value
        # Movielens 100K
        # https://www.researchgate.net/publication/311430762_Extracting_Implicit_Social_Relation_for_Social_Recommendation_Techniques_in_User_Rating_Prediction
        # https://www.researchgate.net/figure/Comparing-MAE-and-RMSE-in-theMovieLens-100K-datasets_fig3_311430762
        # https://www.researchgate.net/publication/335440931_A_Generic_Framework_for_Learning_Explicit_and_Implicit_User-Item_Couplings_in_Recommendation/figures?lo=1

        # Movielens 1M
        # https://www.researchgate.net/publication/358451271_A_hinge-loss_based_codebook_transfer_for_cross-domain_recommendation_with_non-overlapping_data
        # https://www.mdpi.com/2073-431X/10/10/123
        # https://www.researchgate.net/publication/335440931_A_Generic_Framework_for_Learning_Explicit_and_Implicit_User-Item_Couplings_in_Recommendation/figures?lo=1
        # https://www.researchgate.net/publication/303698729_A_Neural_Autoregressive_Approach_to_Collaborative_Filtering/figures?lo=1

        self.__100k_table = pd.DataFrame({
            'Algorithm': ['GlobalAvg', 'UserAvg', 'ItemAvg', 'SlopeOne', 'UserKNN', 'ItemKNN',
                          'RegSVD', 'BiasedMF', 'SVD++1', 'SVD++2', 'Hell-TrustSVD1', 'Hell-TrustSVD2', 'MMMF',
                          'MINDTL', 'TRACER', 'CBT', 'MMMF', 'MINDTL', 'TRACER', 'CBT', 'KNN', 'Co-Clustering'],
            'MAE': [0.944, 0.835, 0.817, 0.738, 0.735, 0.725, 0.733, 0.72, 0.72, 0.719, 0.716, 0.716, 0.6402, 0.7965,
                    0.8039, 0.7746, 0.6808, 1.6538, 0.8213, 0.8239, 0.7732, 0.7582],
            'RMSE': [1.125, 1.042, 1.024, 0.939, 0.942, 0.925, 0.928, 0.914, 0.921, 0.914, 0.909, 0.909, 0.9361, 0.9948,
                     0.9800, 0.9676, 0.9828, 1.9026, 1.0027, 1.0264, 0.979, 0.9675]
        })
        self.__1m_table = pd.DataFrame({
            'Algorithm': ['MMMF', 'MINDTL', 'TRACER', 'CBT', 'BMF', 'DeepFM', 'deepCF', 'lCoupledCF', 'gCoupledCF',
                          'CoupledCF', 'UPCSim', 'CB-UPCSim'],
            'MAE': [0.6389, 1.5984, 0.8145, 0.8725, 0.7929, 0.7452, 0.7264, 0.9462, 0.7467, 0.7285, 0.6993, 0.6857],
            'RMSE': [0.9349, 1.8545, 0.9892, 1.0729, 1.1334, 0.9504, 0.9304, 1.1484, 0.9406, 0.9295, 0.8921, 0.8784]
        })
        if self.__selected_ds is Dataset.movielens_100k.value[0]:
            self.title_text = '100K'
            self.data = self.__100k_table
        elif self.__selected_ds is Dataset.movielens_1m.value[0]:
            self.title_text = '1M'
            self.data = self.__1m_table

    def show_plot(self, algorithm_name=None, mae_value=None, rmse_value=None, save_plot=False):
        if self.__selected_ds is Dataset.movielens_100k.value[0]:
            self.title_text = '100K'
        elif self.__selected_ds is Dataset.movielens_1m.value[0]:
            self.title_text = '1M'

        if algorithm_name is not None and mae_value is not None and rmse_value is not None:
            new_row = pd.DataFrame({'Algorithm': [algorithm_name], 'MAE': [mae_value], 'RMSE': [rmse_value]})
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            self.data.sort_values(by='RMSE', ascending=True, inplace=True)

        x = range(len(self.data))
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3)

        bar_width = 0.35
        ax.bar(x, self.data['MAE'], width=bar_width, align='center', label='MAE')
        ax.bar([p + bar_width for p in x], self.data['RMSE'], width=bar_width, align='center', label='RMSE')

        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Scores')
        ax.set_title('Comparing MAE and RMSE in the MovieLens ' + self.title_text + ' datasets')
        ax.set_xticks([p + bar_width / 2 for p in x])
        ax.set_xticklabels(self.data['Algorithm'], rotation=45, ha='right')
        ax.legend()

        if save_plot:
            plt.savefig('result_' + self.title_text + '.png')

        plt.show()

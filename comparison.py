import pandas as pd
import matplotlib.pyplot as plt
from dataset import Dataset


class Comparison:
    def __init__(self, dataset=Dataset.movielens_1m):
        self.title_text = '100K'
        self.__dataset = dataset
        self.__selected_ds, _ = self.__dataset.value
        # https://www.researchgate.net/publication/311430762_Extracting_Implicit_Social_Relation_for_Social_Recommendation_Techniques_in_User_Rating_Prediction
        # https://www.researchgate.net/figure/Comparing-MAE-and-RMSE-in-theMovieLens-100K-datasets_fig3_311430762
        # https://www.researchgate.net/publication/358451271_A_hinge-loss_based_codebook_transfer_for_cross-domain_recommendation_with_non-overlapping_data
        # https://www.mdpi.com/2073-431X/10/10/123
        #fixme:: this data below belong to 100k also make a similar chart for the 1m dataset
        #https://www.researchgate.net/publication/303698729_A_Neural_Autoregressive_Approach_to_Collaborative_Filtering/figures?lo=1
        #https://www.researchgate.net/publication/335440931_A_Generic_Framework_for_Learning_Explicit_and_Implicit_User-Item_Couplings_in_Recommendation/figures?lo=1
        self.data = pd.DataFrame({
            'Algorithm': ['GlobalAvg', 'UserAvg', 'ItemAvg', 'SlopeOne', 'UserKNN', 'ItemKNN',
                          'RegSVD', 'BiasedMF', 'SVD++1', 'SVD++2', 'Hell-TrustSVD1', 'Hell-TrustSVD2', 'MMMF',
                          'MINDTL', 'TRACER', 'CBT', 'UPCSim', 'CB-UPCSim'],
            'MAE': [0.944, 0.835, 0.817, 0.738, 0.735, 0.725, 0.733, 0.72, 0.72, 0.719, 0.716, 0.716, 0.6402, 0.7965,
                    0.8039, 0.7746, 0.6993, 0.6857],
            'RMSE': [1.125, 1.042, 1.024, 0.939, 0.942, 0.925, 0.928, 0.914, 0.921, 0.914, 0.909, 0.909, 0.9361, 0.9948,
                     0.9800, 0.9676, 0.8921, 0.8784]
        })

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
            plt.savefig('result_'+self.title_text+'.png')

        plt.show()

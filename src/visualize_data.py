import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


class VisualizeData:
    def __init__(self, data):
        self.data = data

    def visualize_claim_by_covertype(self):
        """plots a bar chart for total claim and cover types.
        """
        grouped = self.data.groupby('CoverType')[
            'TotalClaims'].mean().reset_index()

        grouped.plot(kind='bar', x='CoverType', y='TotalClaims',
                     legend=False, figsize=(8, 5))

        plt.title('Average Total Claims by CoverType')
        plt.ylabel('Average Total Claims')
        plt.xlabel('CoverType')
        plt.show()

    def visualize_premiums_by_province(self):
        """Bar chart for premiums by province."""
        grouped = self.data.groupby(
            'Province')['TotalPremium'].mean().reset_index()
        grouped.plot(kind='bar', x='Province', y='TotalPremium',
                     legend=False, figsize=(8, 5))
        plt.title('Average Premiums by Province')
        plt.ylabel('Average Premiums')
        plt.xlabel('Province')
        plt.show()

    def visualize_claims_by_vehicle(self):
        grouped = self.data.groupby('VehicleType')[
            'TotalClaims'].mean().reset_index()
        grouped.plot(kind='bar', x='VehicleType', y='TotalClaims',
                     legend=False, figsize=(8, 5))
        plt.title('Average claim by vehicle type')
        plt.ylabel('Average claim')
        plt.xlabel('vehicle type')
        plt.show()

    def visualize_premium_to_claim_ratio_by_make(self):
        self.data['Premium_to_claim_ratio'] = self.data['TotalPremium'] / \
            (self.data['TotalClaims'] + 1)
        sns.scatterplot(x='make', y='Premium_to_claim_ratio', data=self.data)
        plt.title('Premium-to-Claim Ratio by make')
        plt.ylabel('Premium-to-Claim Ratio')
        plt.xlabel('make')
        plt.show()

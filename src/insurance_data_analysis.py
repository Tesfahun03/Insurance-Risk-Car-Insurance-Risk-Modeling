import pandas as pd
import numpy as np
import matplotlib.pylab as plt


class InsuranceAnalysis:
    def __init__(self, data):
        self.data = data

    def claim_accros_bankand_acc(self):
        """generate descriptive analysis for the average claim accros each bank and account type.
        """
        return self.data.groupby('Bank', 'AccountType').agg(
            avg_claim=('TotalClaims', 'mean'),
            count=('TotalClaims', 'size')
        ).reset_index()

    def claim_accros_covertpye(self):
        return self.data.groupby('CoverType')['TotalClaims'].mean().reset_index()

    def claim_accros_vehicle(self):
        return self.data.groupby('VehicleType', 'Province').agg(
            avg_claim=('TotalClaims', 'mean'),
            avg_Premium=('TotalPremium', 'mean'),
            count=('TotalClaims', 'size')
        ).reset_index()

    def claim_by_gender_province(self):
        return self.data.groupby(['Province', 'Gender']).agg(
            Avg_Total_Claim=('TotalClaim', 'mean'),
            Avg_Premium=('TotalPremium', 'mean'),
            Count=('TotalClaim', 'size')
        ).reset_index()

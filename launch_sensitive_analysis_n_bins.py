from navigation.additional_assessments.sensitive_analysis import SensitiveAnalysis_Bins_Sensors

if __name__ == '__main__':
    pipeline = SensitiveAnalysis_Bins_Sensors('./config_simulations/config_sensitive_analysis.yaml')
    pipeline.start_analysis()

import pytest
import matplotlib.pyplot as plt
from src.plot_metrcis import MetricsVisualizer


@pytest.fixture
def sample_metrics():
    """Fixture to provide sample metrics for plotting."""
    models = ['Linear', 'Tree']
    mae_scores = [1.0, 2.0]
    mse_scores = [2.0, 4.0]
    r2_scores = [0.9, 0.8]
    return models, mae_scores, mse_scores, r2_scores


def test_plot_metrics(sample_metrics):
    models, mae_scores, mse_scores, r2_scores = sample_metrics
    visualizer = MetricsVisualizer(models, mae_scores, mse_scores, r2_scores)
    ax1, ax2, ax3 = visualizer.plot_metrics()
    assert ax1.get_title() == 'Comparison of MAE Scores'
    assert ax2.get_title() == 'Comparison of MSE Scores'
    assert ax3.get_title() == 'Comparison of R2 Scores'
    plt.close('all')

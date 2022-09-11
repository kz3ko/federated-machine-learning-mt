from pandas import read_csv
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator

ONE_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_ITERATIONS_AGGREGATE_1/metrics/' \
                           'server.csv'
THREE_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_ITERATIONS_AGGREGATE_2/metrics/' \
                           'server.csv'
FIVE_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_ITERATIONS_AGGREGATE_3/metrics/' \
                           'server.csv'
SEVEN_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_ITERATIONS_AGGREGATE_5/metrics/' \
                           'server.csv'


def main():
    first_model_metrics = read_csv(ONE_METRICS_PATH)
    second_model_metrics = read_csv(THREE_METRICS_PATH)
    third_model_metrics = read_csv(FIVE_METRICS_PATH)
    fourth_model_metrics = read_csv(SEVEN_METRICS_PATH)

    figure_accuracy = figure(figsize=(6, 5))
    figure_accuracy.set_tight_layout('True')
    axis = figure_accuracy.add_subplot()
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    first_model_line = axis.plot(first_model_metrics['iterations'], first_model_metrics['accuracy'], label='Δi = 1')
    second_model_line = axis.plot(second_model_metrics['iterations'], second_model_metrics['accuracy'], label='Δi = 2')
    third_model_line = axis.plot(third_model_metrics['iterations'], third_model_metrics['accuracy'], label='Δi = 3')
    fourth_model_line = axis.plot(fourth_model_metrics['iterations'], fourth_model_metrics['accuracy'], label='Δi = 5')

    axis.set_title(f'Wykres dokładności (ang. accuracy) modeli globalnych podczas uczenia\n'
                   f' federacyjnego na zbiorze testowym dla różnych\n'
                   f' wartości parametru Δi.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Dokładność modelu')
    axis.grid(True)
    axis.legend(handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line], loc='lower right')

    figure_accuracy.savefig('accuracy_federated_all_global_models_iterations_experiment.jpg', dpi=150)

    figure_loss = figure(figsize=(6, 5))
    figure_loss.set_tight_layout('True')
    axis = figure_loss.add_subplot()
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    first_model_line = axis.plot(first_model_metrics['iterations'], first_model_metrics['loss'], label='Δi = 1')
    second_model_line = axis.plot(second_model_metrics['iterations'], second_model_metrics['loss'], label='Δi = 2')
    third_model_line = axis.plot(third_model_metrics['iterations'], third_model_metrics['loss'], label='Δi = 3')
    fourth_model_line = axis.plot(fourth_model_metrics['iterations'], fourth_model_metrics['loss'], label='Δi = 5')

    axis.set_title(f'Wykres strat (ang. loss) modeli globalnych podczas uczenia\n'
                   f' federacyjnego na zbiorze testowym dla różnych\n'
                   f' wartości parametru Δi.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Wartość funkcji strat')
    axis.grid(True)
    axis.legend(
        handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line], loc='upper right')

    figure_loss.savefig('loss_federated_all_global_models_iterations_experiment.jpg', dpi=150)


if __name__ == '__main__':
    main()

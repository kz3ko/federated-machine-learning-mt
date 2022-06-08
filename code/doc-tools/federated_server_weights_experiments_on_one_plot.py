from pandas import read_csv
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator

ONE_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_WEIGHTS_0_0_0/metrics/' \
                           'server.csv'
THREE_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_WEIGHTS_0_0_1/metrics/' \
                           'server.csv'
FIVE_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_WEIGHTS_0_0_2/metrics/' \
                           'server.csv'
SEVEN_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_WEIGHTS_0_0_3/metrics/' \
                           'server.csv'
NINE_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_WEIGHTS_0_0_5/metrics/' \
                           'server.csv'
ONE_O_METRICS_PATH = '../../docs/generated_data_to_be_used/FEDERATED_EXPERIMENT_WEIGHTS_0_1/metrics/' \
                           'server.csv'


def main():
    first_model_metrics = read_csv(ONE_METRICS_PATH)
    second_model_metrics = read_csv(THREE_METRICS_PATH)
    third_model_metrics = read_csv(FIVE_METRICS_PATH)
    fourth_model_metrics = read_csv(SEVEN_METRICS_PATH)
    fifth_model_metrics = read_csv(NINE_METRICS_PATH)
    sixth_model_metrics = read_csv(ONE_O_METRICS_PATH)

    figure_accuracy = figure(figsize=(6, 5))
    figure_accuracy.set_tight_layout('True')
    axis = figure_accuracy.add_subplot()
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    first_model_line = axis.plot(first_model_metrics['iterations'], first_model_metrics['accuracy'], label='Δw = 0.0')
    second_model_line = axis.plot(second_model_metrics['iterations'], second_model_metrics['accuracy'], label='Δw = 0.001')
    third_model_line = axis.plot(third_model_metrics['iterations'], third_model_metrics['accuracy'], label='Δw = 0.002')
    fourth_model_line = axis.plot(fourth_model_metrics['iterations'], fourth_model_metrics['accuracy'], label='Δw = 0.003')
    fifth_model_line = axis.plot(fifth_model_metrics['iterations'], fifth_model_metrics['accuracy'], label='Δw = 0.005')
    sixth_model_line = axis.plot(sixth_model_metrics['iterations'], sixth_model_metrics['accuracy'], label='Δw = 0.01')

    axis.set_title(f'Wykres dokładności (ang. accuracy) modeli globalnych podczas \nuczenia'
                   f' federacyjnego na zbiorze testowym dla \nróżnych'
                   f' wartości parametru Δw.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Dokładność modelu')
    axis.grid(True)
    axis.legend(handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line, *fifth_model_line, *sixth_model_line], loc='lower right')

    figure_accuracy.savefig('accuracy_federated_all_global_models_weights_experiment.jpg', dpi=150)

    figure_loss = figure(figsize=(6, 5))
    figure_loss.set_tight_layout('True')
    axis = figure_loss.add_subplot()
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    first_model_line = axis.plot(first_model_metrics['iterations'], first_model_metrics['loss'], label='Δw = 0.0')
    second_model_line = axis.plot(second_model_metrics['iterations'], second_model_metrics['loss'], label='Δw = 0.001')
    third_model_line = axis.plot(third_model_metrics['iterations'], third_model_metrics['loss'], label='Δw = 0.002')
    fourth_model_line = axis.plot(fourth_model_metrics['iterations'], fourth_model_metrics['loss'], label='Δw = 0.003')
    fifth_model_line = axis.plot(fifth_model_metrics['iterations'], fifth_model_metrics['loss'], label='Δw = 0.005')
    sixth_model_line = axis.plot(sixth_model_metrics['iterations'], sixth_model_metrics['loss'], label='Δw = 0.01')

    axis.set_title(f'Wykres strat (ang. loss) modeli globalnych podczas uczenia\n'
                   f' federacyjnego na zbiorze testowym dla różnych\n'
                   f' wartości parametru Δw.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Wartość funkcji strat')
    axis.grid(True)
    axis.legend(
        handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line, *fifth_model_line,
                 *sixth_model_line], loc='upper right')

    figure_loss.savefig('loss_federated_all_global_models_weights_experiment.jpg', dpi=150)


if __name__ == '__main__':
    main()

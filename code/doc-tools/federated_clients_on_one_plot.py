from pandas import read_csv
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator

FIRST_MODEL_METRICS_PATH = '../../docs/generated_data_to_be_used/FIRST_MODEL_FEDERATED/metrics/' \
                           'server.csv'
SECOND_MODEL_METRICS_PATH ='../../docs/generated_data_to_be_used/SECOND_MODEL_FEDERATED/metrics/' \
                           'server.csv'
THIRD_MODEL_METRICS_PATH ='../../docs/generated_data_to_be_used/THIRD_MODEL_FEDERATED/metrics/' \
                           'server.csv'
FOURTH_MODEL_METRICS_PATH ='../../docs/generated_data_to_be_used/FOURTH_MODEL_FEDERATED/metrics/' \
                           'server.csv'


def main():
    first_model_metrics = read_csv(FIRST_MODEL_METRICS_PATH)
    second_model_metrics = read_csv(SECOND_MODEL_METRICS_PATH)
    third_model_metrics = read_csv(THIRD_MODEL_METRICS_PATH)
    fourth_model_metrics = read_csv(FOURTH_MODEL_METRICS_PATH)

    figure_accuracy = figure(figsize=(6, 5))
    figure_accuracy.set_tight_layout('True')
    axis = figure_accuracy.add_subplot()
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    first_model_line = axis.plot(first_model_metrics['iterations'], first_model_metrics['accuracy'], label='Model pierwszy')
    second_model_line = axis.plot(second_model_metrics['iterations'], second_model_metrics['accuracy'], label='Model drugi')
    third_model_line = axis.plot(third_model_metrics['iterations'], third_model_metrics['accuracy'], label='Model trzeci')
    fourth_model_line = axis.plot(fourth_model_metrics['iterations'], fourth_model_metrics['accuracy'], label='Model czwarty')

    axis.set_title(f'Wykres dokładności (ang. accuracy) modeli globalnych podczas \nuczenia federacyjnego na zbiorze testowym.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Dokładność modelu')
    axis.grid(True)
    axis.legend(handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line], loc='lower right')

    figure_accuracy.savefig('accuracy_federated_all_global_models.jpg', dpi=150)


if __name__ == '__main__':
    main()

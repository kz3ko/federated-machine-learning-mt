from pandas import read_csv
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator

FIRST_MODEL_METRICS_PATH = '../../docs/generated_data_to_be_used/FIRST_MODEL_TRADITIONAL/metrics/' \
                           'traditional_participant_training_metrics.csv'
SECOND_MODEL_METRICS_PATH ='../../docs/generated_data_to_be_used/SECOND_MODEL_TRADITIONAL/metrics/' \
                           'traditional_participant_training_metrics.csv'
THIRD_MODEL_METRICS_PATH ='../../docs/generated_data_to_be_used/THIRD_MODEL_TRADITIONAL/metrics/' \
                           'traditional_participant_training_metrics.csv'
FOURTH_MODEL_METRICS_PATH ='../../docs/generated_data_to_be_used/FOURTH_MODEL_TRADITIONAL/metrics/' \
                           'traditional_participant_training_metrics.csv'


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

    axis.set_title(f'Wykres dokładności (ang. accuracy) modeli podczas uczenia\n tradycyjnego na zbiorze uczącym.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Dokładność modelu')
    axis.grid(True)
    axis.legend(handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line], loc='upper left')

    figure_accuracy.savefig('accuracy_traditional_all_models.jpg', dpi=150)

    figure_loss = figure(figsize=(6, 5))
    figure_loss.set_tight_layout('True')
    axis = figure_loss.add_subplot()
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    first_model_line = axis.plot(first_model_metrics['iterations'], first_model_metrics['loss'], label='Model pierwszy')
    second_model_line = axis.plot(second_model_metrics['iterations'], second_model_metrics['loss'], label='Model drugi')
    third_model_line = axis.plot(third_model_metrics['iterations'], third_model_metrics['loss'], label='Model trzeci')
    fourth_model_line = axis.plot(fourth_model_metrics['iterations'], fourth_model_metrics['loss'], label='Model czwarty')

    axis.set_title(f'Wykres strat (ang. loss) modeli podczas uczenia\n tradycyjnego na zbiorze uczącym.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Wartosc funkcji strat')
    axis.grid(True)
    axis.legend(handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line], loc='upper right')

    figure_loss.savefig('loss_traditional_all_models.jpg', dpi=150)



    figure_val_accuracy = figure(figsize=(6, 5))
    figure_val_accuracy.set_tight_layout('True')
    axis = figure_val_accuracy.add_subplot()
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    first_model_line = axis.plot(first_model_metrics['iterations'], first_model_metrics['val_accuracy'], label='Model pierwszy')
    second_model_line = axis.plot(second_model_metrics['iterations'], second_model_metrics['val_accuracy'], label='Model drugi')
    third_model_line = axis.plot(third_model_metrics['iterations'], third_model_metrics['val_accuracy'], label='Model trzeci')
    fourth_model_line = axis.plot(fourth_model_metrics['iterations'], fourth_model_metrics['val_accuracy'], label='Model czwarty')

    axis.set_title(f'Wykres dokładności (ang. accuracy) modeli podczas uczenia\n tradycyjnego na zbiorze walidacyjnym.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Dokładność modelu')
    axis.grid(True)
    axis.legend(handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line], loc='upper left')

    figure_val_accuracy.savefig('val_accuracy_traditional_all_models.jpg', dpi=150)


    figure_val_loss = figure(figsize=(6, 5))
    figure_val_loss.set_tight_layout('True')
    axis = figure_val_loss.add_subplot()
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    first_model_line = axis.plot(first_model_metrics['iterations'], first_model_metrics['val_loss'], label='Model pierwszy')
    second_model_line = axis.plot(second_model_metrics['iterations'], second_model_metrics['val_loss'], label='Model drugi')
    third_model_line = axis.plot(third_model_metrics['iterations'], third_model_metrics['val_loss'], label='Model trzeci')
    fourth_model_line = axis.plot(fourth_model_metrics['iterations'], fourth_model_metrics['val_loss'], label='Model czwarty')

    axis.set_title(f'Wykres strat (ang. loss) modeli podczas uczenia\n tradycyjnego na zbiorze walidacyjnym.')
    axis.set_xlabel('Iteracje')
    axis.set_ylabel('Wartosc funkcji strat')
    axis.grid(True)
    axis.legend(handles=[*first_model_line, *second_model_line, *third_model_line, *fourth_model_line], loc='upper right')

    figure_val_loss.savefig('val_loss_traditional_all_models.jpg', dpi=150)


if __name__ == '__main__':
    main()

from analytics.plotter import Plotter, ClientLearningPlotter, ServerTestingPlotter, \
    TraditionalParticipantLearningPlotter, ConfusionMatrixMaker
from analytics.metrics_collector import MetricsCollector
from learning.participant import Participants, Client, Server, TraditionalParticipant


class PlotterCreator:

    def __init__(self, participants: Participants, metrics_collector: MetricsCollector):
        self.participants = participants
        self.metrics_collector = metrics_collector

    def create_plotters(self) -> list[Plotter]:
        plotters = []
        for participant in self.participants:
            if isinstance(participant, Client):
                plotters.append(ClientLearningPlotter(self.metrics_collector))
            elif isinstance(participant, Server):
                plotters.append(ServerTestingPlotter(self.metrics_collector))
            elif isinstance(participant, TraditionalParticipant):
                plotters.append(TraditionalParticipantLearningPlotter(self.metrics_collector))

        plotters.append(ConfusionMatrixMaker(self.metrics_collector))

        return plotters

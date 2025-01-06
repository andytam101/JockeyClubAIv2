from dataloader.participation_ranking_loader import ParticipationRankingLoader

dataloader_dict = {
    "PRL": ParticipationRankingLoader,
}


def load_dataloader(dl_name):
    dataloader = dataloader_dict[dl_name]
    return dataloader()

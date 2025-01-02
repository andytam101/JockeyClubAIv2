from dataloader.simple_loader import SimpleLoader

dataloader_dict = {
    "SimpleLoader": SimpleLoader,
}


def load_dataloader(dl_name):
    dataloader = dataloader_dict[dl_name]
    return dataloader()

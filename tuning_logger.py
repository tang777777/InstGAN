import logging
import os


def setup_logging(loss_file, param_file):

    # create log directories if they don't exist
    os.makedirs(os.path.dirname(loss_file), exist_ok=True)
    os.makedirs(os.path.dirname(param_file), exist_ok=True)
    
    # set up logging configuration for loss file
    loss_logger = logging.getLogger('loss_logger')
    loss_logger.setLevel(logging.INFO)
    loss_handler = logging.FileHandler(loss_file)
    loss_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    loss_logger.addHandler(loss_handler)

    # set up logging configuration for param file
    param_logger = logging.getLogger('param_logger')
    param_logger.setLevel(logging.INFO)
    param_handler = logging.FileHandler(param_file)
    param_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    param_logger.addHandler(param_handler)
    
    return loss_logger, param_logger

def log_loss(loss_logger, param_logger, step, generator_loss, discriminator_loss, entropy):
    # log generator and discriminator loss in loss file
    loss_logger.info(f"Step {step} - Generator loss: {generator_loss:.4f}, Discriminator loss: {discriminator_loss:.4f}")

    # log other parameter in param file
    param_logger.info(f"Step {step} - Entropy: {entropy:.4f}")
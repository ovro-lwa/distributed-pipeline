import casatasks as ct
import logging

logging.basicConfig(level=logging.INFO)

def list_obs(ms: str) -> None:
    logging.info(ct.listobs(ms))

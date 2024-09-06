import yaml
import logging

logger = logging.getLogger(__name__)


def load_model_configs(config_path="model_configs.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

        # Validate the configuration
        if (
            "models" not in config
            or not isinstance(config["models"], dict)
            or not config["models"]
        ):
            raise ValueError(
                "Invalid configuration file: 'models' section is missing or empty."
            )

        keep_model_loaded = config.get("keep_model_loaded", True)
        model_unload_delay_secs = config.get("model_unload_delay_secs", 0)
        default_model_name = config.get("default_model")

        if not default_model_name:
            logger.warning(
                "No default model specified. Loading the first model from the list."
            )
            model_names = list(config["models"].keys())
            if model_names:
                default_model_name = model_names[0]
            else:
                raise ValueError("No models found in the configuration file.")

        model_configs = config["models"]

        return default_model_name, model_configs, keep_model_loaded, model_unload_delay_secs

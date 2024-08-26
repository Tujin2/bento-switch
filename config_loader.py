import yaml
import logging

logger = logging.getLogger(__name__)


def load_model_configs(config_path="model_configs.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        default_model_name = config["models"].get("default_model")

        # Validate the configuration
        if "models" not in config or not isinstance(config["models"], dict) or not config["models"]:
            raise ValueError("Invalid configuration file: 'models' section is missing or empty.")

        if not default_model_name:
            logger.warning(
                "No default model specified. Loading the first model from the list."
            )
            model_names = [name for name in config["models"] if name != "default_model"]
            if model_names:
                default_model_name = model_names[0]
            else:
                raise ValueError("No models found in the configuration file.")

        model_defaults = {
            model_name: model_config.get("default_params", {})
            for model_name, model_config in config["models"].items()
        }

        return default_model_name, model_defaults

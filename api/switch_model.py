from fastapi import HTTPException
from bentoml import api
from models.exceptions import ModelNotFoundException, ModelLoadException


@api(route="/switch_model")
async def switch_model(self, model_name: str):
    try:
        self.model_manager.switch_model(model_name)
        return {"message": f"Successfully switched to model: {model_name}"}
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelLoadException as e:
        raise HTTPException(status_code=500, detail=str(e))

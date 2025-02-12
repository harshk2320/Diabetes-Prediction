from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import DiabetesData, DiabetesDataClassifier
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the diabetes-related attributes expected from the form.
    """
    def __init__(self, request: Request):

        self.request: Request = request
        self.Pregnancies: Optional[int] = None
        self.Glucose: Optional[float] = None
        self.BloodPressure: Optional[float] = None
        self.SkinThickness: Optional[float] = None
        self.Insulin: Optional[float] = None
        self.BMI: Optional[float] = None
        self.DiabetesPedigreeFunction: Optional[float] = None
        self.Age: Optional[int] = None


    async def get_diabetes_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.Pregnancies = form.get("Pregnancies")
        self.Glucose = form.get("Glucose")
        self.BloodPressure = form.get("BloodPressure")
        self.SkinThickness = form.get("SkinThickness")
        self.Insulin = form.get("Insulin")
        self.BMI = form.get("BMI")
        self.DiabetesPedigreeFunction = form.get("DiabetesPedigreeFunction")
        self.Age = form.get("Age")

# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for diabetes data input.
    """
    return templates.TemplateResponse(
            "diabetesdata.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_diabetes_data()
        
        diabetes_data = DiabetesData(
                                Pregnancies= form.Pregnancies,
                                Glucose = form.Glucose,
                                BloodPressure = form.BloodPressure,
                                SkinThickness = form.SkinThickness,
                                Insulin = form.Insulin,
                                BMI = form.BMI,
                                DiabetesPedigreeFunction = form.DiabetesPedigreeFunction,
                                Age = form.Age
                                )

        # Convert form data into a DataFrame for the model
        diabetes_df = diabetes_data.get_diabetes_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = DiabetesDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=diabetes_df)[0]

        # Interpret the prediction result as 'Response-Yes' or 'Response-No'
        status = "Response-Yes" if value == 1 else "Response-No"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "diabetesdata.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
















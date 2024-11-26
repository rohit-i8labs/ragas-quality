import plotly.graph_objects as go
import plotly.io as pio
import os
import uuid
from datasets import Dataset
from ragas import evaluate
from dotenv import load_dotenv
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
load_dotenv()

# Create the 'images' folder if it doesn't exist
if not os.path.exists('static/images'):
    os.makedirs('static/images')

def evaluate_rag(test_set:dict):
    # Convert sample dataset to Dataset object
    dataSet = Dataset.from_dict(test_set)

    # Evaluate the dataset using Ragas metrics
    result = evaluate(
        dataSet,
        metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_correctness,
        answer_similarity
        ],
    )

    df = result.to_pandas()

    print("==========================")
    print("Plotting Graph")

    # Prepare the data
    data = {
        'context_precision': result['context_precision'],
        'faithfulness': result['faithfulness'],
        'answer_relevancy': result['answer_relevancy'],
        'context_recall': result['context_recall'],
        'answer_correctness': result['answer_correctness'],
        'answer_similarity': result['answer_similarity'],
    }

    # Create the radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(data.values()),
        theta=list(data.keys()),
        fill='toself',
        name='Ensemble RAG'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Retrieval Augmented Generation - Evaluation',
        width=800,
    )

    # Generate a unique file name using uuid
    unique_filename = f"image_{uuid.uuid4().hex}.png"
    image_path = os.path.join('static/images', unique_filename)

    # Save the plot as a PNG image
    pio.write_image(fig, image_path, format='png')

    print(f"Graph saved at: {image_path}")

    # Return the image path
    return(result,df,image_path)
# Use Miniconda (Python 3.8 is included)
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file into the container
COPY env/environment.yml /app/env/environment.yml

# Install dependencies from environment.yml
RUN conda env create -f /app/env/environment.yml

# Copy the rest of the code into the container
COPY . /app

# Run the training script inside the Conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "transformer_env", "python", "train.py"]

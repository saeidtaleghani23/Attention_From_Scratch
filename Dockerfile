# Use Miniconda (Python 3.8 is included)
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file into the container
COPY env/environment.yml /app/env/environment.yml

# Install dependencies from environment.yml
RUN conda env create -f /app/env/environment.yml

# Activate the environment and install any additional dependencies if needed
RUN echo "conda activate transformer_env" >> ~/.bashrc

# Ensure the Conda environment is activated correctly
SHELL ["/bin/bash", "--login", "-c"]

# Copy the rest of the code into the container
COPY . /app

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port Flask will run on
EXPOSE 5000

# Run the Flask application inside the Conda environment
#CMD ["conda", "run", "--no-capture-output", "-n", "transformer_env", "flask", "run"]
CMD ["bash", "-c", "source activate transformer_env && flask run --host=0.0.0.0"]

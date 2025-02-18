# Step 1: Use an official Python base image from Docker Hub
FROM python:3.10-slim-buster

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements.txt (or any dependency file) into the container
COPY requirements.txt .

# Step 4: Install dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the project into the container
COPY . /app

# Step 6: Expose the port that Flask will use
EXPOSE 5000

# Step 7: Set the command to run the application when the container starts
CMD ["python", "src/app.py"]

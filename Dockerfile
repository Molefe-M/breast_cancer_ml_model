# Step 1: Use an official Python base image from Docker Hub
FROM python:3.10-slim-bullseye AS base

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the requirements.txt (or any dependency file) into the container
COPY requirements.txt .

# Step 4: Install dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# Step 5: Reuse base image and add dedicated non-root user
FROM base
RUN adduser --system --uid 1000 --group appuser

# Step 6: Copy the rest of the project into the container
COPY . /app

# Step 7: Ensure 'appuser' owns the /app directory and switch to appuser
RUN chown -R appuser:appuser /app
USER appuser

# Step 8: Expose the port that Flask will use
EXPOSE 5000

# Set the environment variable for Flask to run in production mode
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Step 9: Set the command to run the application when the container starts
CMD ["python", "app.py"]

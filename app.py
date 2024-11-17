from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
from logistic_regression import do_experiments

app = Flask(__name__)

# Route to render the main page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the experiment logic
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        # Parse input parameters
        data = request.json
        start = float(data.get('start'))
        end = float(data.get('end'))
        step_num = int(data.get('step_num'))

        # Run the experiment and retrieve generated filenames
        dataset_file, parameters_file = do_experiments(start, end, step_num)

        # Generate URLs for the output images
        dataset_url = url_for('static', filename=dataset_file)
        parameters_url = url_for('static', filename=parameters_file)

        # Debug prints (can be removed in production)
        app.logger.info(f"Dataset Image URL: {dataset_url}")
        app.logger.info(f"Parameters Image URL: {parameters_url}")

        # Return URLs as JSON response
        return jsonify({
            "dataset_img": dataset_url,
            "parameters_img": parameters_url
        })

    except Exception as e:
        app.logger.error(f"Error running experiment: {e}")
        return jsonify({"error": "An error occurred while running the experiment."}), 500

# Route to serve result images directly
@app.route('/results/<path:filename>')
def serve_results(filename):
    try:
        return send_from_directory('results', filename)
    except FileNotFoundError:
        app.logger.error(f"File not found: {filename}")
        return jsonify({"error": "File not found."}), 404

if __name__ == '__main__':
    app.run(debug=True)

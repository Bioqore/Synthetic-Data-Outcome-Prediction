#!/usr/bin/env python3
"""
Flask Web App for Synthetic Outcome Prediction
=============================================

This app provides a simple web interface for:
- Uploading a training CSV (historical data)
- Uploading an input CSV (rows needing predicted outcomes)
- Running the Gaussian-Copula + surrogate model pipeline
- Displaying a preview and allowing download of the results CSV
"""

import os
import io
import uuid
import tempfile
import pandas as pd
from flask import Flask, request, redirect, url_for, send_file, render_template_string, flash

from main_pipeline import PipelineConfig, SyntheticDataPipeline

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Synthetic Outcome Prediction</title>
    <style>
      body { font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 2rem; }
      .container { max-width: 900px; margin: 0 auto; }
      .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
      .btn { background: #2563eb; color: #fff; border: none; padding: 0.6rem 1rem; border-radius: 6px; cursor: pointer; }
      .btn:disabled { background: #93c5fd; cursor: not-allowed; }
      .row { display: flex; gap: 1rem; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; font-size: 14px; }
      th { background: #f3f4f6; text-align: left; }
      .alert { background: #fef3c7; border: 1px solid #fde68a; color: #92400e; padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Synthetic Outcome Prediction</h1>
      {% with messages = get_flashed_messages() %}
        {% if messages %}
          <div class="alert">{{ messages[0] }}</div>
        {% endif %}
      {% endwith %}

      <div class="card">
        <form action="{{ url_for('predict') }}" method="post" enctype="multipart/form-data">
          <div class="row">
            <div style="flex:1">
              <label>Training CSV (historical data)</label>
              <input type="file" name="training_file" accept=".csv" required />
            </div>
            <div style="flex:1">
              <label>Input CSV (rows needing outcomes)</label>
              <input type="file" name="input_file" accept=".csv" required />
            </div>
          </div>
          <div style="margin: 0.75rem 0;">
            <label>
              <input type="checkbox" name="augment" /> Use synthetic data augmentation
            </label>
          </div>
          <button type="submit" class="btn">Run Prediction</button>
        </form>
      </div>

      {% if result_preview is defined %}
        <div class="card">
          <h3>Results (preview)</h3>
          <div>{{ result_preview|safe }}</div>
          <div style="margin-top: 0.75rem;">
            <a class="btn" href="{{ url_for('download', filename=download_name) }}">Download Results CSV</a>
          </div>
        </div>
      {% endif %}
    </div>
  </body>
</html>
"""

# In-memory map of download tokens to absolute file paths
_DOWNLOADS = {}


def _save_upload(file_storage, prefix: str) -> str:
    temp_dir = tempfile.gettempdir()
    ext = os.path.splitext(file_storage.filename or "")[1].lower() or ".csv"
    unique_name = f"{prefix}_{uuid.uuid4().hex}{ext}"
    out_path = os.path.join(temp_dir, unique_name)
    file_storage.save(out_path)
    return out_path


@app.get("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.post("/predict")
def predict():
    try:
        if "training_file" not in request.files or "input_file" not in request.files:
            flash("Please upload both training and input CSV files.")
            return redirect(url_for("index"))

        training_fs = request.files["training_file"]
        input_fs = request.files["input_file"]

        if training_fs.filename == "" or input_fs.filename == "":
            flash("Both files must have a filename.")
            return redirect(url_for("index"))

        # Save uploads to temp
        training_path = _save_upload(training_fs, "training")
        input_path = _save_upload(input_fs, "input")

        # Output file in temp dir using input name stem
        temp_dir = tempfile.gettempdir()
        input_stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(temp_dir, f"{input_stem}_results.csv")

        # Load dataframes to infer schema
        try:
            train_df = pd.read_csv(training_path)
            pred_df = pd.read_csv(input_path)
        except Exception as e:
            flash(f"Failed to read CSV files: {e}")
            return redirect(url_for("index"))

        # Determine id column if present (optional convenience)
        id_column = None
        if "plasmid_id" in pred_df.columns:
            id_column = "plasmid_id"

        # Infer target columns
        # 1) Try common two-target schemas
        target_columns = None
        multi_target_options = [
            ["yield_score", "burden_score"],
            ["Yield", "Burden"],
            ["yield", "burden"],
        ]
        for pair in multi_target_options:
            if all(col in train_df.columns for col in pair):
                target_columns = list(pair)
                break
        # 2) Fall back to single-target schemas
        if target_columns is None:
            single_target_options = ["yield_score", "Yield", "yield"]
            for col in single_target_options:
                if col in train_df.columns:
                    target_columns = [col]
                    break
        if target_columns is None:
            # 3) Generic fallback: assume last column of training is the target
            if len(train_df.columns) >= 2:
                last_col = train_df.columns[-1]
                # Avoid using id column as target if mis-ordered
                if last_col != id_column:
                    target_columns = [last_col]
        if target_columns is None:
            flash("Could not infer target columns. Ensure the final column in the training CSV is the outcome (e.g., 'Yield').")
            return redirect(url_for("index"))

        # Determine required feature columns from training (all except targets and optional id)
        train_feature_columns = [c for c in train_df.columns if c not in target_columns and c != id_column]
        # Verify prediction CSV contains all required feature columns
        missing_features = [c for c in train_feature_columns if c not in pred_df.columns]
        if missing_features:
            flash(f"Prediction CSV is missing required feature columns: {missing_features}")
            return redirect(url_for("index"))

        # Infer categorical columns from object dtype among training features that are present in both files
        cat_candidates = train_df[train_feature_columns].select_dtypes(include=["object"]).columns.tolist()
        cat_columns = [c for c in cat_candidates if c in pred_df.columns]
        # Fallback to common plasmid fields if none detected
        if not cat_columns:
            common_plasmid = ["promoter", "ori", "antibiotic", "rbs", "poi", "terminator"]
            cat_columns = [c for c in common_plasmid if c in train_df.columns and c in pred_df.columns]
        if not cat_columns:
            flash("Could not infer categorical feature columns shared by training and input CSVs.")
            return redirect(url_for("index"))

        # Configure pipeline
        config = PipelineConfig(
            historical_data_path=training_path,
            prediction_input_path=input_path,
            output_path=output_path,
            id_column=id_column,
            target_columns=target_columns,
            categorical_columns=cat_columns,
        )

        pipeline = SyntheticDataPipeline(config)

        # Validate inputs
        validation = pipeline.validate_pipeline()
        if not validation.get("overall_valid", False):
            issues = "; ".join(validation.get("issues", []))
            flash(f"Validation failed: {issues}")
            return redirect(url_for("index"))

        # Run with or without augmentation
        use_augmentation = bool(request.form.get("augment"))
        if use_augmentation:
            results_df = pipeline.run_with_synthetic_augmentation()
        else:
            results_df = pipeline.run_pipeline()

        # Create a small HTML preview
        preview_html = results_df.head(50).to_html(index=False)

        # Register downloadable file by tokenized name
        download_name = f"results_{uuid.uuid4().hex}.csv"
        _DOWNLOADS[download_name] = output_path

        return render_template_string(
            HTML_TEMPLATE,
            result_preview=preview_html,
            download_name=download_name,
        )

    except Exception as e:
        flash(f"Error: {str(e)}")
        return redirect(url_for("index"))


@app.get("/download/<filename>")
def download(filename: str):
    # Resolve token to actual path and stream
    path = _DOWNLOADS.get(filename)
    if not path or not os.path.exists(path):
        flash("File not found or expired.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

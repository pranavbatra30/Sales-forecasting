import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
import os
import csv
from flask import Response
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['UPLOAD_FOLDER'] = 'uploads'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    global output
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            test_data = pd.read_csv(filepath)
            test_data['date'] = pd.to_datetime(test_data['date'])
            last_date = test_data['date'].max()

            # Generate a sequence of dates starting from the next month
            forecast_start_date = pd.to_datetime(last_date) + pd.DateOffset(months=1)
            forecast_dates = pd.date_range(forecast_start_date, periods=18, freq='MS')

            csv_file_path = "/Users/pranavbatra/my_flask_project/data.csv"
            data = pd.read_csv(csv_file_path)
            # Aggregate sales data to a monthly level
            data['date'] = pd.to_datetime(data['fiscal_year_historical'].astype(str) + '-' + data['fiscal_month_historical'].astype(str) + '-01')
            data_monthly = data.groupby(['date', 'business_unit_group_name', 'company_region_name_level_1', 'product_line_code', 'product_line_name'], as_index=False).sum()

            # List of business units
            business_units = ['Channel - Industrial', 'Appliances', 'Data and Devices', 'Energy', 'Industrial', 'Industrial Commercial Transportation']

            def train_and_evaluate_model(business_unit, data):
                # Filter the data for the business unit and create an explicit copy
                data_filtered = data[data['business_unit_group_name'] == business_unit].copy()

                # Encode categorical columns
                labelencoder_region = LabelEncoder()
                labelencoder_code = LabelEncoder()
                labelencoder_name = LabelEncoder()
                data_filtered['company_region_name_level_1'] = labelencoder_region.fit_transform(data_filtered['company_region_name_level_1'])
                data_filtered['product_line_code'] = labelencoder_code.fit_transform(data_filtered['product_line_code'])
                data_filtered['product_line_name'] = labelencoder_name.fit_transform(data_filtered['product_line_name'])

                # Create dummy variables for categorical columns
                onehotencoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                categorical_columns = ['company_region_name_level_1', 'product_line_code', 'product_line_name']
                onehot_data = onehotencoder.fit_transform(data_filtered[categorical_columns])
                onehot_df = pd.DataFrame(onehot_data, columns=onehotencoder.get_feature_names_out(categorical_columns), index=data_filtered.index)
                data_filtered = pd.concat([data_filtered.drop(categorical_columns, axis=1), onehot_df], axis=1)

                # Split the data into training and testing sets
                X = data_filtered.drop(['sales_amount', 'business_unit_group_name', 'date'], axis=1)
                y = data_filtered['sales_amount']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model = HistGradientBoostingRegressor()
                model.fit(X_train, y_train)

                # Test the model
                y_pred = model.predict(X_test)

                # Evaluate the model
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_pred, y_test)
                return model, r2, mae, labelencoder_region, labelencoder_code, labelencoder_name, onehotencoder
            # Initialize an empty DataFrame to store the output
            output = pd.DataFrame(columns=['Date', 'Business Unit', 'Region', 'Product_line_code', 'Forecast value'])

            # Train and evaluate a model for each business unit
            models = {}
            encoders = {}
            for business_unit in business_units:
                    model, r2, mae, X_test, y_test, y_pred, encoders_bu = train_and_evaluate_model(business_unit, data_monthly)
                    models[business_unit] = model
                    encoders[business_unit] = encoders_bu
                    print(f"Model score for {business_unit}: Accuracy: {r2}")

                    # Get the recent 18 months of data
                    recent_18_months = data_monthly[(data_monthly['business_unit_group_name'] == business_unit)].tail(18).reset_index(drop=True)

                    # Create dummy variables for categorical columns
                    categorical_columns = ['company_region_name_level_1', 'product_line_code', 'product_line_name']
                    onehot_data = encoders_bu.transform(recent_18_months[categorical_columns])
                    onehot_df = pd.DataFrame(onehot_data, columns=encoders_bu.get_feature_names_out(categorical_columns), index=recent_18_months.index)
                    recent_18_months_encoded = pd.concat([recent_18_months.drop(categorical_columns, axis=1), onehot_df], axis=1)

                    # Prepare the input data for prediction
                    recent_18_months_X = recent_18_months_encoded.drop(['sales_amount', 'business_unit_group_name', 'date'], axis=1)
                    recent_18_months_y = recent_18_months_encoded['sales_amount']

                    # Make predictions
                    recent_18_months_y_pred = model.predict(recent_18_months_X)

                    # Populate the output DataFrame with the required information
                    temp_output = recent_18_months_encoded.copy()
                    temp_output['Date'] = forecast_dates.strftime('%Y-%m-%d')
                    temp_output['Business Unit'] = business_unit
                    temp_output['Region'] = recent_18_months['company_region_name_level_1']
                    temp_output['Product_line_code'] = recent_18_months['product_line_code']
                    temp_output['Forecast value'] = recent_18_months_y_pred

                    # Drop unnecessary columns
                    temp_output = temp_output[['Date', 'Business Unit', 'Region', 'Product_line_code', 'Forecast value']]

                    # Append the temporary output to the main output DataFrame
                    output = pd.concat([output, temp_output], ignore_index=True)

            return render_template('index.html', forecast=output.to_dict(orient='records'), business_units=business_units)

    return render_template('index.html')
    
@app.route('/download', methods=['GET'])
def download():
    global output

    # Convert forecast data to CSV format
    csv_data = output.to_csv(index=False)

    # Create a response with the CSV data and proper headers
    response = Response(
        csv_data,
        content_type='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=forecast.csv',
            'Content-Type': 'text/csv; charset=utf-8'
        }
    )

    return response

@app.route('/forecast', methods=['POST'])
def forecast():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .api import fetch_stock_data
from .backtests import backtest_strategy, calculate_metrics
from .ML import predict_prices, calculate_prediction_metrics, generate_stock_price_plot
from .reports import generate_pdf_report

# Create your views here.
def update_stock_data(request, symbol):
    try:
        fetch_stock_data(symbol=symbol)
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'Status': 'Error', 'Message': str(e)}, status=500)
    
def backtest_view(request):
    symbol = request.GET.get('symbol', 'AAPL') 
    initial_investment = float(request.GET.get('initial_investment', 10000))
    short_window = int(request.GET.get('short_window', 50))
    long_window = int(request.GET.get('long_window', 200))

    result = backtest_strategy(symbol=symbol, initial_investment=initial_investment, short_window=short_window, long_window=long_window)
    return JsonResponse(result)

def predict_view(request):
    symbol = request.GET.get('symbol', 'AAPL') 
    days = int(request.GET.get('days', 30)) 

    predictions = predict_prices(symbol=symbol, days=days)

    return JsonResponse({
        'symbol': symbol,
        'Predictions': predictions.tolist()
    })

def report_view(request, symbol):
    format = request.GET.get('format', 'json')
    
    # Perform backtest and prediction
    backtest_results = backtest_strategy(symbol=symbol)
    predictions = predict_prices(symbol=symbol)
    
    # Calculate metrics
    backtest_metrics = calculate_metrics(backtest_results)
    prediction_metrics = calculate_prediction_metrics(
        actual_prices=backtest_results['actual_prices'],
        predicted_prices=predictions
    )
    
    # Combine metrics into a report
    report_data = {
        'backtest_metrics': backtest_metrics,
        'prediction_metrics': prediction_metrics,
    }
    
    if format == 'json':
        # Return report as JSON
        return JsonResponse(report_data)
    else:
        # Generate PDF
        generate_stock_price_plot(
            actual_prices=backtest_results['actual_prices'], 
            predicted_prices=predictions, 
            dates=backtest_results['dates'], 
            symbol=symbol
        )
        pdf_filename = generate_pdf_report(report_data, symbol)
        
        # Return PDF as response
        with open(pdf_filename, 'rb') as pdf_file:
            response = HttpResponse(pdf_file.read(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{pdf_filename}"'
            return response
        

def generate_report(request):
    symbol = request.GET.get('symbol', 'AAPL')
    days = request.GET.get('days', 30)  # Default to 30 days if not provided

    # Perform backtest and prediction to get metrics
    backtest_results = backtest_strategy(symbol=symbol, initial_investment=10000)
    predictions = predict_prices(symbol=symbol)

    # Calculate metrics (modify as needed based on your actual metric calculation functions)
    backtest_metrics = calculate_metrics(backtest_results)
    prediction_metrics = calculate_prediction_metrics(
        actual_prices=backtest_results['actual_prices'],
        predicted_prices=predictions
    )

    # Combine metrics
    metrics = {
        'Backtest Metrics': backtest_metrics,
        'Prediction Metrics': prediction_metrics,
    }

    # Generate PDF report
    pdf_filename = generate_pdf_report(symbol, days, metrics)
    return JsonResponse({'file': pdf_filename})
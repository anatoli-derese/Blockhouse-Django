from django.urls import path
from . import views

urlpatterns = [
    path('update/<str:symbol>/', views.update_stock_data, name='update_stock_data'),
    path('backtest/', views.backtest_view),
    path('predict/', views.predict_view, name="predict_prices"),
    path('report/', views.generate_report, name='stock_report'),
]
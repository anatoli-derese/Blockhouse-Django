from django.db import models

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    close_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    high_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    low_price = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    volume = models.BigIntegerField()
    is_predicted = models.BooleanField(default=False) 

    class Meta:
        unique_together = ('symbol', 'date')
        ordering = ['-date']

    def __str__(self):
        return f'{self.symbol} - {self.date}'
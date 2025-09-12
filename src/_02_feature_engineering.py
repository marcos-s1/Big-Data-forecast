from datetime import datetime, timedelta

def calculate_week_of_month(transaction_date, reference_date):
    """
    Calculates the week number within the month for a given transaction date.

    Args:
        transaction_date (date): The date of the transaction (YYYY-MM-DD).
        reference_date (date): A reference date within the month (YYYY-MM-DD).

    Returns:
        int: The week number within the month (starting from 1).
    """
    # Ensure dates are datetime objects
    if isinstance(transaction_date, str):
        transaction_date = datetime.strptime(transaction_date, '%Y-%m-%d').date()
    if isinstance(reference_date, str):
        reference_date = datetime.strptime(reference_date, '%Y-%m-%d').date()

    # Convert reference_date to the first day of the month
    first_day_of_month = reference_date.replace(day=1)

    # Calculate the difference in days
    day_difference = (transaction_date - first_day_of_month).days

    # Calculate the week number (starting from 1)
    week_number = (day_difference // 7) + 1

    return week_number


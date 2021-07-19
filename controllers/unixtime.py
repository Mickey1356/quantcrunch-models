  
from datetime import datetime
from pytz import timezone

fmt = "%Y-%m-%d %H:%M:%S"

def get_time_SG():
    return datetime.now(timezone('Singapore'))

def get_time_NY():
    return datetime.now(timezone('America/New_York'))

def convert_NY_to_SG(dt):
    old_timezone = timezone("America/New_York")
    new_timezone = timezone("Singapore")
    localized_timestamp = old_timezone.localize(dt)
    new_timezone_timestamp = localized_timestamp.astimezone(new_timezone)
    return new_timezone_timestamp

def convert_NY_str_to_SG_str(str):
    nydt = datetime.strptime(str, fmt)
    sgdt = convert_NY_to_SG(nydt)
    return datetime.strftime(sgdt, fmt)

def convert_UTC_to_NY(dt):
    """
    Note that GMT time is UTC time. 
    """
    old_timezone = timezone("GMT")
    new_timezone = timezone("America/New_York")
    localized_timestamp = old_timezone.localize(dt)
    new_timezone_timestamp = localized_timestamp.astimezone(new_timezone)
    return new_timezone_timestamp

def unix_to_NY(unix_time):
    """
    unix_time is in milliseconds
    returns datetime object, in NY time
    """
    unix_time /= 1000
    dt = datetime.utcfromtimestamp(unix_time)
    return convert_UTC_to_NY(dt)

def is_dst(tz, datetime_to_check):
    """Determine whether or not Daylight Savings Time (DST)
    is currently in effect"""

    # Jan 1 of this year, when all tz assumed to not be in dst
    non_dst = datetime(year=datetime.now().year, month=1, day=1)
    # Make time zone aware based on tz passed in
    non_dst_tz_aware = timezone(tz).localize(non_dst)

    # if DST is in effect, their offsets will be different
    return not (non_dst_tz_aware.utcoffset() == datetime_to_check.utcoffset())

def is_dst_NY_now():
    now = get_time_NY()
    return is_dst('America/New_York', now)
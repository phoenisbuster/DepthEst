# DepthEst-App
 Server source code


(Option 1)
!!! Run Flask + WSGI (ubuntu)

$ gunicorn server_api:app -w 4 -b 0.0.0.0:6969




(Option 2)
!!! Run Flask + Celery + Redis

1. Start the Celery worker:
> celery -A server_celery.celery worker --loglevel=info --pool=solo  

2. Run Flask server_celery.py
from django.conf.urls import url,include
from .views import *
from django.contrib.auth import views
from django.contrib.auth.views import *

urlpatterns=[
    url(r'^$', home),       
    url(r'^home/$', home, name='home'),
    url(r'^submitcausedata/$', submitCauseData, name='submitCauseData'),    
]

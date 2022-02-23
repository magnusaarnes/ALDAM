from django.http import HttpResponse
from django.shortcuts import render
from django.views import generic
# Create your views here.


class HomeView(generic.View):
    template = 'saliency_gui/home_view.html'
    
    def get(self, request):
        return render(request, self.template)
        

    def post(self, request):
        return render(request, self.template)
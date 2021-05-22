from django.db import models

# Create your models here.
class image(models.Model):
    image = models.BinaryField(blank = True)
    target = models.IntegerField('target')

    class meta:
        ordering = ['target']
    def __str__(self,):
        return 'an image'


class Author(models.Model):
    """Model representing an author."""
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    date_of_birth = models.DateField(null=True, blank=True)
    date_of_death = models.DateField('Died', null=True, blank=True)

    class Meta:
        ordering = ['last_name', 'first_name']

    def __str__(self):
        """String for representing the Model object."""
        return f'{self.last_name}, {self.first_name}'


from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("recognition", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="faceembedding",
            name="vector",
        ),
        migrations.AddField(
            model_name="faceembedding",
            name="encrypted_vector",
            field=models.TextField(default=""),
            preserve_default=False,
        ),
    ]

import pusher



def sent_notification(channel: str, event_name, data: dict)->None:
    pusher_client = pusher.Pusher(
    app_id='1710242',
    key='ddc62434eb2ed7d73396',
    secret='b1b62f64e6e2d7e31865',
    cluster='us2',
    ssl=True
    )

    pusher_client.trigger(channel, event_name, data)
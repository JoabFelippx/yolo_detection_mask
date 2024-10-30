from is_wire.core import Logger, Subscription
from streamChannel import StreamChannel

import cv2
import numpy as np

from utils import draw_masks, to_np, load_model, unpack_message, draw_boxes
import json

def main():
    
    config = json.load(open('etc/config/config.json'))

    log = Logger(name='StreamConsumer')

    broker_uri = f'amqp://{config["broker"]["username"]}:{config["broker"]["password"]}@{config["broker"]["host"]}:{config["broker"]["port"]}'
    topic = config['broker']['topic']['subscribe']

    channel = StreamChannel(uri=broker_uri)
    subscription = Subscription(channel)
    subscription.subscribe(topic=topic)

    log.info(f'StreamConsumer is listening to topic {topic}')

    model = load_model(config['model']['path'])
    
    log.info(f'Model {config["model"]["path"]} loaded')

    while True:

        message = channel.consume()

        if type(message) == bool:
            log.info(f'No message received from topic {topic}')
            continue

        log.info(f'Message received from topic {topic}')

        image = unpack_message(message)
        image_np = to_np(image)

        results = model.predict(image_np, conf=config['model']['confidence'], classes=config['model']['classes'])

        image_np = draw_masks(image_np, results)
        image_np = draw_boxes(image_np, results)

        cv2.imshow("Image", image_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
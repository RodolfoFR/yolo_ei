import socket
from is_wire.core import Channel


# Essa classe herdar propriedades da classe Channel (de is_wire.core), que serve para comunicação
class StreamChannel(Channel):

    # o init dele é igual ao init da classe Channel
    def __init__(self, uri="amqp://guest:guest@localhost:5672", exchange="is"):
        super().__init__(uri=uri, exchange=exchange)

   
    def consume_last(self, return_dropped=False):
        dropped = 0
        # consume a mensagem igual a classe Channel
        msg = super().consume()
        # fica rodando até da dar erro de conexão
        while True:
            try:
                # will raise an exceptin when no message remained
                msg = super().consume(timeout=0.0) # consume
                dropped += 1 # aumenta dropped
               
            # para se tiver problemas de conecção o server, se passar tempo demais
            except socket.timeout:
                return (msg, dropped) if return_dropped else msg
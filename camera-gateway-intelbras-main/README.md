Serviço - Câmera Gateway
==================

O serviço Camera Gateway (CG) é dividido em duas partes: gateway e driver. O driver é responsável pelo controle, através do recebimento de requisições, das funcionalidades das câmeras. As funcionalidades são: estabelecimento de conexão com a câmera; captura de transmissão de imagem em tempo real; definição e obtenção do tipo da imagem capturada; definição e obtenção do número de quadros por segundo (FPS) da câmera; definição e obtenção da resolução da câmera; O driver é acessível somente por formas indiretas, isto é, requisições vindas do gateway.

O gateway, por sua vez, é responsável pela controle e exposição da imagem da câmera, através do driver. O gateway, paralelamente, publica uma imagem capturada e permanece aguardando um pedido de criação de requisição. Quando um pedido de requisição é feito por um usuário ou serviço, é criado, então, uma requisição com uma mensagem do tipo is_msg.CameraConfig para obter ou definir alguma configuração da câmera. Em seguida, através do driver, a requisição é encaminhada para câmera. Após a ação da câmera, é encaminhada uma resposta da requisição, e, em seguida, o gateway retorna para o estado de aguardo de pedido de requisição.

Dependências:
-----
Este serviço não possui dependências em relação a outros serviços.

Eventos:
--------
<img width=850/> ⇒ Mensagem recebida | <img width=850/> Encaminha ⇒ | <img width=500/> Descrição  
:------------ | :-------- | :----------
:incoming_envelope: **tópico:** `CameraGateway.{cameragateway_id}.SetConfig` <br> :gem: **tipo de mensagem:** [CameraConfig] | :incoming_envelope: **tópico:** `{request.reply_to}` <br> :gem: **tipo de mensagem:** [Empty] | `[RPC] Define as configurações da câmera`
:incoming_envelope: **tópico:** `CameraGateway.{cameragateway_id}.GetConfig` <br> :gem: **tipo de mensagem:** [FieldSelector] | :incoming_envelope: **tópico:** `{request.reply_to}` <br> :gem: **tipo de mensagem::** [CameraConfig] | `[RPC] Retorna as configurações da câmera`

[CameraConfig]: https://github.com/labviros/is-msgs/tree/master/docs#is.vision.CameraConfig
[Empty]: https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/empty.proto
[FieldSelector]: https://github.com/labviros/is-msgs/tree/master/docs#is.common.FieldSelector

```
{cameragateway_id} = número identificador da câmera, passado no arquivo de configuração.
```

Configuração:
----------------
A configuração do serviço é realizada através de um arquivo JSON, passado como argumento. Um exemplo de configuração pode ser encontrado em: [`etc/conf/options.json`]

Exemplos:
------------
Um exemplo, para captura de imagens da câmera através deste serviço, pode ser encontrado em [`examples/client.py`](examples/client.py).

Um outro exemplo, para modificação dos parâmetros da câmera e para retorno da configuração atual, pode ser encontrado em [`examples/rpc_client.py`](examples/rpc_client.py).

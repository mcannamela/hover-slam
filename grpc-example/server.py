import random
from concurrent import futures
import time

import grpc

import example_pb2
import example_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

PIECES = ['K', 'Q', 'R', 'B', 'N', '']
RANKS = list(range(1, 9))
FILES = 'abcdefgh'


def random_element(x):
    return x[random.randint(0, len(x)-1)]


def random_move():
    return example_pb2.Move(
        piece=random_element(PIECES),
        file=random_element(FILES),
        rank=random_element(RANKS),
    )


class ChessPropServicer(example_pb2_grpc.ChessPropServicer):

    def PointCounterpoint(self, request_iterator, context):
        for move in request_iterator:
            reply = random_move()
            judgement = random.random()
            yield example_pb2.ReplyWithJudgement(
                reply=reply,
                judgement=judgement
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    example_pb2_grpc.add_ChessPropServicer_to_server(
        ChessPropServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("Ready to start judging chess moves!")
    server.start()
    print("Server's up!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()

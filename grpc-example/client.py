import random

import grpc

import example_pb2
import example_pb2_grpc
import time

PIECES = ['K', 'Q', 'R', 'B', 'N', '']
RANKS = list(range(1, 9))
FILES = list('abcdefgh')


def random_element(x):
    return x[random.randint(0, len(x)-1)]


def random_move():
    return example_pb2.Move(
        piece=random_element(PIECES),
        file=random_element(FILES),
        rank=random_element(RANKS),
    )

def generate_random_moves():
    while True:
        move = random_move()
        print("Propose: {}{}{}".format(move.piece, move.file, move.rank))
        time.sleep(1)
        yield move


def make_moves(stub):
    responses = stub.PointCounterpoint(generate_random_moves())
    for response in responses:
        move = response.reply
        judgement = response.judgement
        print("Reply:  {}{}{}, {}".format(move.piece, move.file, move.rank, judgement))


def run():
    channel = grpc.insecure_channel('chess-prop:50051')
    stub = example_pb2_grpc.ChessPropStub(channel)
    print("-------------- PointCounterpoint --------------")
    make_moves(stub)


if __name__ == '__main__':
    run()
import pygame
from pong import Game
import neat  
import os 
import pickle

class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, net):
        run = True
        clock = pygame.time.Clock()

        while run:
            clock.tick(60)
            game_info = self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            # user moves the paddle
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            # ai move paddle
            output = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x-self.ball.x)))
            decision = output.index(max(output))

            if decision == 1:
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:
                self.game.move_paddle(left=False, up=False)

            # print(game_info.left_score, game_info.right_score)
            self.game.draw(True, False)
            pygame.display.update()


    def train_ai(self, genome1, genome2, config):
        # create neural nets
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        while run:
            # quit game if user exits
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            output1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x-self.ball.x)))
            decision1 = output1.index(max(output1))
            
            # movements for paddle 1 (left)
            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            elif decision1 == 2:
                self.game.move_paddle(left=True, up=False)

            output2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x-self.ball.x)))
            decision2 = output2.index(max(output2))

            # movements for paddle 2 (right)
            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            elif decision2 == 2:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.loop()
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break
        
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits



def eval_genomes(genomes, config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    # train over each pair of AIs
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break

        genome1.fitness = 0
        for (genome_id2, genome2) in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)


def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-xx')
    p = neat.Population(config)

    # see data on the screen
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # run fitness function and get winner
    winner = p.run(eval_genomes, 50)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

def test_ai(config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    game = PongGame(window, width, height)
    game.test_ai(winner_net)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    run_neat(config)
    # test_ai(config)
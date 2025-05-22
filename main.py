import pygame 
import chess
from Engine import AlphaZeroEngine
from random import choice

# * Configuration
SQUARE_SIZE     = 80
NUM_SQUARES     = 8
BOARD_WIDTH     = SQUARE_SIZE * NUM_SQUARES
BOARD_HEIGHT    = SQUARE_SIZE * NUM_SQUARES

PANEL_WIDTH     = 400
PANEL_MARGIN    = 0.1 * PANEL_WIDTH

WIDTH, HEIGHT   = BOARD_WIDTH + PANEL_WIDTH, BOARD_HEIGHT

# * Color
WHITE     = (234, 234, 210)
BLACK     = (75, 114, 153)
PANEL_BG  = (50, 50, 50)
BTN_COLOR = (100, 200, 100)
BTN_HOVER = (120, 220, 120)
TEXT_COL  = (255, 255, 255)

# * Piece
PIECE_SCALE = 0.7
PIECE_SIZE  = int(SQUARE_SIZE * PIECE_SCALE)
MARGIN      = (SQUARE_SIZE - PIECE_SIZE) // 2

start_btn_width = PANEL_WIDTH - PANEL_MARGIN * 2
start_btn_height = 50

BOX_WIDTH = 500
BOX_HEIGHT = 300

move_delay = 500

def play_chess(board: chess.Board, agent: AlphaZeroEngine, agent_side):
    turn = "W"
    
    while True:
        if board.is_game_over():
            outcome = board.outcome()
            
            if outcome.winner is True:
                result = "White win"
            elif outcome.winner is False:
                result = "Black win"
            else:
                result = f"Draw by {outcome.termination.name}"

            yield "GAME_OVER", result

        if turn == agent_side:
            move = engine.predict(board)
        else:
            move = choice(list(board.legal_moves))
            
        board.push(move)
        yield "MOVE", move
        turn = "B" if turn=="W" else "W"

        
def load_images(path):
    images = {}
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    colors = ['w', 'b']
    
    for color in colors:
        for piece in pieces:
            sym = piece if color == 'b' else piece.upper()
            image_path = f"{path}/{color}_{piece}.png"
            img = pygame.image.load(image_path).convert_alpha()
            
            images[sym] = pygame.transform.smoothscale(img, (PIECE_SIZE, PIECE_SIZE))

    return images

def draw_piece(screen, board: chess.Board, images):
    for square, piece in board.piece_map().items():
        x, y = chess.square_file(square), chess.square_rank(square)
        px, py = x * SQUARE_SIZE + MARGIN, (NUM_SQUARES - 1 - y) * SQUARE_SIZE + MARGIN
    
        screen.blit(images[piece.symbol()], (px, py))

def draw_button(screen, mx, my, btn, font, text):
    color = BTN_HOVER if btn.collidepoint(mx, my) else BTN_COLOR
    pygame.draw.rect(screen, color, btn)
    txt = font.render(text, True, TEXT_COL)
    screen.blit(txt, (btn.x + (start_btn_width - txt.get_width()) // 2, btn.y + (start_btn_height - txt.get_height()) // 2))

if __name__ == "__main__":
    # * Pygame init
    pygame.init()
    screen  = pygame.display.set_mode((WIDTH, HEIGHT))
    clock   = pygame.time.Clock()
    font    = pygame.font.SysFont('Ubuntu', 24)

    # * Board and piece init
    img_path = "./assets/img"
    board = chess.Board()
    images = load_images(img_path)

    # * Init engine
    param_path = "params.pth"
    engine = AlphaZeroEngine(param_path)

    # * Flag
    running_state = True
    begin_state = True
    choose_side_state = False
    restart_state = False
    game_start = False

    # * Params
    agent_side = None

    # * Button init
    start_btn_rect = pygame.Rect(BOARD_WIDTH + PANEL_MARGIN , (HEIGHT - start_btn_height) // 2, start_btn_width, start_btn_height)
    white_side_btn = pygame.Rect(BOARD_WIDTH + PANEL_MARGIN , HEIGHT // 2 - start_btn_height, start_btn_width, start_btn_height)
    black_side_btn = pygame.Rect(BOARD_WIDTH + PANEL_MARGIN , HEIGHT // 2 + start_btn_height, start_btn_width, start_btn_height)
    restart_btn = pygame.Rect(BOARD_WIDTH + PANEL_MARGIN , (HEIGHT - start_btn_height) // 2, start_btn_width, start_btn_height)    

    # * Handle double click
    last_click_time = 0
    CLICK_COOLDOWN = 200  # ms

    while running_state:
        mx, my = pygame.mouse.get_pos()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running_state = False

            # * Onclick event
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                now = pygame.time.get_ticks()
                if now - last_click_time < CLICK_COOLDOWN:
                    continue

                last_click_time = now

                # * Start state -> Choose side state
                if start_btn_rect.collidepoint(mx, my) and begin_state:
                    begin_state = False
                    choose_side_state = True
                    continue
                
                # * Choose side state -> Play state
                if choose_side_state:
                    for btn, name in [(white_side_btn, "W"), (black_side_btn, "B")]:
                        if btn.collidepoint(mx, my):
                            agent_side = name
                            choose_side_state = False

                            game_start = True
                            stepper = play_chess(board, engine, agent_side)

                            last_move_time = pygame.time.get_ticks()

                # * Game done - Restart state => Start State
                if restart_state and restart_btn.collidepoint(mx, my):
                    restart_state = False
                    begin_state = True

                    board = chess.Board() # * Restart new game
                    agent_side = None

        # * Engine render
        if game_start and pygame.time.get_ticks() - last_move_time >= move_delay:
            try:
                tag, payload = next(stepper)
                if tag == "MOVE":
                    pass
                elif tag == "GAME_OVER":
                    game_over_msg = payload
                    game_start = False
                    restart_state = True

                last_move_time = pygame.time.get_ticks()

            except StopIteration:
                started = False

        # * Draw side panel 
        pygame.draw.rect(screen, PANEL_BG, (BOARD_WIDTH, 0, PANEL_WIDTH, HEIGHT))
        
        # * Chess board 
        for r in range(NUM_SQUARES):
            for c in range(NUM_SQUARES):
                color = WHITE if (r + c) % 2 == 0 else BLACK
                rect = (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(screen, color, rect)

        # * Draw base on state
        # 1. Draw start button
        if begin_state:
            draw_button(screen, mx, my, start_btn_rect, font, "Start")
        
        # 2. Draw choose side button
        if choose_side_state:
            draw_button(screen, mx, my, white_side_btn, font, "White")
            draw_button(screen, mx, my, black_side_btn, font, "Black")

        # 3. Draw message and restart button
        if restart_state:
            txt = font.render(str(game_over_msg), True, TEXT_COL)

            x = BOARD_WIDTH + (PANEL_WIDTH - txt.get_width()) // 2
            y = restart_btn.y - 100  
            
            screen.blit(txt, (x, y))
            draw_button(screen, mx, my, restart_btn, font, "Restart")

        # *  Draw pieces
        draw_piece(screen, board, images)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
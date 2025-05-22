from NNArchitechure import *
import chess

class AlphaZeroEngine:
    def __init__(self, path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniChessNet().to(self.device)
        # * Load tune parameters to model
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
    
        self.direction_table = [
            (+1, 0),  # N
            (+1,+1),  # NE
            ( 0,+1),  # E
            (-1,+1),  # SE
            (-1, 0),  # S
            (-1,-1),  # SW
            ( 0,-1),  # W
            (+1,-1),  # NW
        ]

        # 8 knight moves (dr, dc)
        self.knight_moves = [
            (+2,+1), (+1,+2), (-1,+2), (-2,+1),
            (-2,-1), (-1,-2), (+1,-2), (+2,-1),
        ]

        self.promo_order = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

        self.ACTION_PLANES = 73

    def sign(self, x: int) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)
    
    def board_to_planes(self, board: chess.Board):
        planes = torch.zeros((18, 8, 8), dtype=torch.float32, device=self.device)

        piece_to_plane = {
            (chess.PAWN, True) : 0,
            (chess.KNIGHT, True) : 1,
            (chess.BISHOP, True) : 2,
            (chess.ROOK, True) : 3,
            (chess.QUEEN, True) : 4,
            (chess.KING, True) : 5,
            (chess.PAWN, False) : 6,
            (chess.KNIGHT, False) : 7,
            (chess.BISHOP, False) : 8,
            (chess.ROOK, False) : 9,
            (chess.QUEEN, False): 10,
            (chess.KING, False) : 11,
        }

        # Piece plane
        for square, piece in board.piece_map().items():
            row = chess.square_rank(square)
            col = chess.square_file(square)
            plane_idx = piece_to_plane[(piece.piece_type, piece.color)]
            planes[plane_idx, row, col] = 1

        # Side to move plan
        if board.turn == chess.WHITE:
            planes[12, :, :] = 1

        # Castling rights
        planes[13, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
        planes[14, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
        planes[15, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
        planes[16, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))

        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            planes[17, :, ep_file] = 1

        return planes.unsqueeze(0).to(self.device)

    def encode_move(self, move: chess.Move) -> int:
        fr = move.from_square               # 0..63
        to = move.to_square
        dr = chess.square_rank(to) - chess.square_rank(fr)
        dc = chess.square_file(to) - chess.square_file(fr)

        if (dr, dc) in self.knight_moves:
            plane = 56 + self.knight_moves.index((dr, dc))

        elif move.promotion and move.promotion != chess.QUEEN:
            pidx = self.promo_order.index(move.promotion)
            if dc == 0:
                t = 0
            elif dc < 0:
                t = 1
            else:
                t = 2
            plane = 56 + 8 + (pidx * 3 + t)

        else:
            d = self.direction_table.index((self.sign(dr), self.sign(dc)))
            step = max(abs(dr), abs(dc))  # 1..7
            plane = d * 7 + (step - 1)

        return fr * self.ACTION_PLANES + plane  # 0..4671


    def decode_move(self, idx: int) -> chess.Move:
        fr = idx // self.ACTION_PLANES
        plane = idx % self.ACTION_PLANES

        rank_fr = chess.square_rank(fr)
        file_fr = chess.square_file(fr)

        # 1) Queen-like moves (0..55)
        if plane < 56:
            d = plane // 7
            step = (plane % 7) + 1
            dr, dc = self.direction_table[d]
            rank_to = rank_fr + dr * step
            file_to = file_fr + dc * step
            to = chess.square(file_to, rank_to)
            return chess.Move(fr, to)

        # 2) Knight jumps (56..63)
        if plane < 56 + 8:
            k = plane - 56
            dr, dc = self.knight_moves[k]
            rank_to = rank_fr + dr
            file_to = file_fr + dc
            to = chess.square(file_to, rank_to)
            return chess.Move(fr, to)

        # 3) Under-promotions (64..72)
        up_plane = plane - (56 + 8)
        pidx = up_plane // 3
        t = up_plane % 3
        promo = self.promo_order[pidx]

        dc = 0 if t == 0 else (-1 if t == 1 else +1)
        if rank_fr == 6:
            # white pawn
            dr = +1
        else:
            # black pawn
            dr = -1
        rank_to = rank_fr + dr
        file_to = file_fr + dc
        to = chess.square(file_to, rank_to)
        return chess.Move(fr, to, promotion=promo)
    
    def predict(self, board: chess.Board) -> chess.Move:
        x = self.board_to_planes(board)               

        with torch.no_grad():
            logits, _ = self.model(x)                 
            probs = torch.softmax(logits[0], dim=-1) 

        action_size = probs.numel()                  
        raw_idxs   = [self.encode_move(m) for m in board.legal_moves]
        legal_idxs = [i for i in raw_idxs if 0 <= i < action_size]

        idx_tensor = torch.tensor(legal_idxs, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(probs)                
        mask.scatter_(0, idx_tensor, 1.0)             

        masked = probs * mask

        best_idx = int(masked.argmax().item())

        return self.decode_move(best_idx)
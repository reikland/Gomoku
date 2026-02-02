#include <SFML/Graphics.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr int kBoardSize = 15;
constexpr int kWinLength = 5;
constexpr int kCellSize = 36;
constexpr int kPadding = 24;
constexpr int kBoardPixels = kCellSize * (kBoardSize - 1);
constexpr int kWindowWidth = kBoardPixels + 2 * kPadding;
constexpr int kWindowHeight = kWindowWidth + 80;
constexpr int kAiMaxDepth = 4;
constexpr int kMoveRadius = 2;
constexpr int kInfinity = 1'000'000;

enum class Player : int { None = 0, Black = 1, White = 2 };

struct Move {
    int x;
    int y;
};

struct HashEntry {
    int depth;
    int score;
    int flag;
};

class Zobrist {
  public:
    Zobrist() {
        std::mt19937_64 rng(42);
        for (auto &row : table_) {
            for (auto &cell : row) {
                cell[0] = rng();
                cell[1] = rng();
            }
        }
    }

    std::uint64_t value(int x, int y, Player player) const {
        if (player == Player::Black) {
            return table_[y][x][0];
        }
        return table_[y][x][1];
    }

  private:
    std::array<std::array<std::array<std::uint64_t, 2>, kBoardSize>, kBoardSize> table_{};
};

class GomokuBoard {
  public:
    GomokuBoard() { clear(); }

    void clear() {
        for (auto &row : board_) {
            row.fill(Player::None);
        }
        moves_.clear();
        hash_ = 0;
    }

    bool isInside(int x, int y) const { return x >= 0 && y >= 0 && x < kBoardSize && y < kBoardSize; }

    Player at(int x, int y) const { return board_[y][x]; }

    bool isEmpty(int x, int y) const { return at(x, y) == Player::None; }

    void place(int x, int y, Player player, const Zobrist &zobrist) {
        board_[y][x] = player;
        moves_.push_back({x, y});
        hash_ ^= zobrist.value(x, y, player);
    }

    void undo(const Zobrist &zobrist) {
        auto move = moves_.back();
        Player player = at(move.x, move.y);
        board_[move.y][move.x] = Player::None;
        moves_.pop_back();
        hash_ ^= zobrist.value(move.x, move.y, player);
    }

    bool hasWinner(Move lastMove, Player player) const {
        if (lastMove.x < 0) {
            return false;
        }
        constexpr int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (auto &dir : directions) {
            int count = 1;
            for (int step = 1; step < kWinLength; ++step) {
                int nx = lastMove.x + dir[0] * step;
                int ny = lastMove.y + dir[1] * step;
                if (!isInside(nx, ny) || at(nx, ny) != player) {
                    break;
                }
                ++count;
            }
            for (int step = 1; step < kWinLength; ++step) {
                int nx = lastMove.x - dir[0] * step;
                int ny = lastMove.y - dir[1] * step;
                if (!isInside(nx, ny) || at(nx, ny) != player) {
                    break;
                }
                ++count;
            }
            if (count >= kWinLength) {
                return true;
            }
        }
        return false;
    }

    const std::vector<Move> &moves() const { return moves_; }

    std::uint64_t hash() const { return hash_; }

    int stonesCount() const { return static_cast<int>(moves_.size()); }

  private:
    std::array<std::array<Player, kBoardSize>, kBoardSize> board_{};
    std::vector<Move> moves_;
    std::uint64_t hash_{};
};

class GomokuAI {
  public:
    GomokuAI() : rng_(1337) {}

    Move bestMove(GomokuBoard &board, Player aiPlayer, std::chrono::milliseconds timeLimit) {
        nodes_ = 0;
        deadline_ = std::chrono::steady_clock::now() + timeLimit;
        bestMove_ = {-1, -1};
        lastCompletedDepth_ = 0;

        for (int depth = 1; depth <= kAiMaxDepth; ++depth) {
            if (std::chrono::steady_clock::now() >= deadline_) {
                break;
            }
            int score = negamax(board, depth, -kInfinity, kInfinity, aiPlayer, aiPlayer);
            if (std::chrono::steady_clock::now() >= deadline_) {
                break;
            }
            lastScore_ = score;
            lastCompletedDepth_ = depth;
        }

        if (bestMove_.x == -1) {
            auto candidates = generateMoves(board);
            if (!candidates.empty()) {
                bestMove_ = candidates.front();
            } else {
                bestMove_ = {kBoardSize / 2, kBoardSize / 2};
            }
        }
        return bestMove_;
    }

    std::int64_t nodes() const { return nodes_; }

    int lastCompletedDepth() const { return lastCompletedDepth_; }

    int lastScore() const { return lastScore_; }

  private:
    int negamax(GomokuBoard &board, int depth, int alpha, int beta, Player player, Player aiPlayer) {
        if (std::chrono::steady_clock::now() >= deadline_) {
            return 0;
        }

        ++nodes_;

        auto lastMove = board.moves().empty() ? Move{-1, -1} : board.moves().back();
        if (lastMove.x != -1 && board.hasWinner(lastMove, opposite(player))) {
            return -kInfinity + (kAiMaxDepth - depth);
        }

        if (depth == 0) {
            return evaluate(board, aiPlayer);
        }

        auto hash = board.hash();
        auto it = transposition_.find(hash);
        if (it != transposition_.end() && it->second.depth >= depth) {
            const auto &entry = it->second;
            if (entry.flag == 0) {
                return entry.score;
            }
            if (entry.flag == -1 && entry.score <= alpha) {
                return entry.score;
            }
            if (entry.flag == 1 && entry.score >= beta) {
                return entry.score;
            }
        }

        auto moves = generateMoves(board);
        if (moves.empty()) {
            return 0;
        }

        orderMoves(board, moves, player);

        int bestScore = -kInfinity;
        int originalAlpha = alpha;
        for (const auto &move : moves) {
            board.place(move.x, move.y, player, zobrist_);
            int score = -negamax(board, depth - 1, -beta, -alpha, opposite(player), aiPlayer);
            board.undo(zobrist_);

            if (score > bestScore) {
                bestScore = score;
                if (depth == lastCompletedDepth_ + 1) {
                    bestMove_ = move;
                }
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                break;
            }
        }

        HashEntry entry{depth, bestScore, 0};
        if (bestScore <= originalAlpha) {
            entry.flag = -1;
        } else if (bestScore >= beta) {
            entry.flag = 1;
        }
        transposition_[hash] = entry;
        return bestScore;
    }

    int evaluate(const GomokuBoard &board, Player aiPlayer) {
        int score = 0;
        for (int y = 0; y < kBoardSize; ++y) {
            for (int x = 0; x < kBoardSize; ++x) {
                Player player = board.at(x, y);
                if (player == Player::None) {
                    continue;
                }
                int value = patternScore(board, x, y, player);
                if (player == aiPlayer) {
                    score += value;
                } else {
                    score -= value;
                }
            }
        }
        return score;
    }

    int patternScore(const GomokuBoard &board, int x, int y, Player player) {
        constexpr int directions[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        int score = 0;
        for (auto &dir : directions) {
            int count = 1;
            int openEnds = 0;
            for (int step = 1; step < kWinLength; ++step) {
                int nx = x + dir[0] * step;
                int ny = y + dir[1] * step;
                if (!board.isInside(nx, ny)) {
                    break;
                }
                if (board.at(nx, ny) == player) {
                    ++count;
                } else {
                    if (board.at(nx, ny) == Player::None) {
                        ++openEnds;
                    }
                    break;
                }
            }
            for (int step = 1; step < kWinLength; ++step) {
                int nx = x - dir[0] * step;
                int ny = y - dir[1] * step;
                if (!board.isInside(nx, ny)) {
                    break;
                }
                if (board.at(nx, ny) == player) {
                    ++count;
                } else {
                    if (board.at(nx, ny) == Player::None) {
                        ++openEnds;
                    }
                    break;
                }
            }
            if (count >= kWinLength) {
                score += 100000;
            } else if (count == 4 && openEnds == 2) {
                score += 10000;
            } else if (count == 4 && openEnds == 1) {
                score += 2000;
            } else if (count == 3 && openEnds == 2) {
                score += 500;
            } else if (count == 3 && openEnds == 1) {
                score += 150;
            } else if (count == 2 && openEnds == 2) {
                score += 40;
            }
        }
        return score;
    }

    std::vector<Move> generateMoves(const GomokuBoard &board) {
        std::vector<Move> moves;
        if (board.moves().empty()) {
            moves.push_back({kBoardSize / 2, kBoardSize / 2});
            return moves;
        }

        std::array<std::array<bool, kBoardSize>, kBoardSize> visited{};
        for (const auto &move : board.moves()) {
            for (int dy = -kMoveRadius; dy <= kMoveRadius; ++dy) {
                for (int dx = -kMoveRadius; dx <= kMoveRadius; ++dx) {
                    int nx = move.x + dx;
                    int ny = move.y + dy;
                    if (!board.isInside(nx, ny) || visited[ny][nx] || !board.isEmpty(nx, ny)) {
                        continue;
                    }
                    visited[ny][nx] = true;
                    moves.push_back({nx, ny});
                }
            }
        }
        return moves;
    }

    void orderMoves(GomokuBoard &board, std::vector<Move> &moves, Player player) {
        std::vector<std::pair<int, Move>> scored;
        scored.reserve(moves.size());
        for (const auto &move : moves) {
            board.place(move.x, move.y, player, zobrist_);
            int score = evaluate(board, player);
            board.undo(zobrist_);
            scored.push_back({score, move});
        }
        std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });
        for (std::size_t i = 0; i < scored.size(); ++i) {
            moves[i] = scored[i].second;
        }
    }

    Player opposite(Player player) {
        return player == Player::Black ? Player::White : Player::Black;
    }

    std::mt19937 rng_;
    Zobrist zobrist_;
    std::unordered_map<std::uint64_t, HashEntry> transposition_;
    std::chrono::steady_clock::time_point deadline_{};
    std::int64_t nodes_{};
    int lastCompletedDepth_{};
    int lastScore_{};
    Move bestMove_{-1, -1};
};

class GomokuGame {
  public:
    GomokuGame()
        : window_(sf::VideoMode(kWindowWidth, kWindowHeight), "Gomoku"),
          fontLoaded_(font_.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
        window_.setFramerateLimit(60);
    }

    void run() {
        while (window_.isOpen()) {
            processEvents();
            draw();
        }
    }

  private:
    void processEvents() {
        sf::Event event{};
        while (window_.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window_.close();
            } else if (event.type == sf::Event::MouseButtonPressed) {
                handleMouseClick(event.mouseButton.x, event.mouseButton.y);
            } else if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::R) {
                    reset();
                }
            }
        }
    }

    void handleMouseClick(int px, int py) {
        if (winner_.has_value()) {
            return;
        }
        auto cell = pixelToCell(px, py);
        if (!cell) {
            return;
        }
        if (!board_.isEmpty(cell->x, cell->y)) {
            return;
        }
        board_.place(cell->x, cell->y, Player::Black, zobrist_);
        if (board_.hasWinner(board_.moves().back(), Player::Black)) {
            winner_ = Player::Black;
            return;
        }
        if (board_.stonesCount() == kBoardSize * kBoardSize) {
            winner_ = Player::None;
            return;
        }
        auto start = std::chrono::steady_clock::now();
        Move aiMove = ai_.bestMove(board_, Player::White, std::chrono::milliseconds(900));
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        lastAiTimeMs_ = duration.count();
        if (board_.isEmpty(aiMove.x, aiMove.y)) {
            board_.place(aiMove.x, aiMove.y, Player::White, zobrist_);
            if (board_.hasWinner(board_.moves().back(), Player::White)) {
                winner_ = Player::White;
            }
        }
    }

    void reset() {
        board_.clear();
        winner_.reset();
        lastAiTimeMs_ = 0;
    }

    std::optional<Move> pixelToCell(int px, int py) {
        int boardTop = kPadding;
        int boardLeft = kPadding;
        if (px < boardLeft - kCellSize / 2 || py < boardTop - kCellSize / 2) {
            return std::nullopt;
        }
        int x = (px - boardLeft + kCellSize / 2) / kCellSize;
        int y = (py - boardTop + kCellSize / 2) / kCellSize;
        if (!board_.isInside(x, y)) {
            return std::nullopt;
        }
        return Move{x, y};
    }

    void draw() {
        window_.clear(sf::Color(245, 224, 176));
        drawGrid();
        drawStones();
        drawHud();
        window_.display();
    }

    void drawGrid() {
        for (int i = 0; i < kBoardSize; ++i) {
            int offset = i * kCellSize;
            sf::Vertex vertical[] = {
                sf::Vertex(sf::Vector2f(kPadding + offset, kPadding), sf::Color::Black),
                sf::Vertex(sf::Vector2f(kPadding + offset, kPadding + kBoardPixels), sf::Color::Black),
            };
            sf::Vertex horizontal[] = {
                sf::Vertex(sf::Vector2f(kPadding, kPadding + offset), sf::Color::Black),
                sf::Vertex(sf::Vector2f(kPadding + kBoardPixels, kPadding + offset), sf::Color::Black),
            };
            window_.draw(vertical, 2, sf::Lines);
            window_.draw(horizontal, 2, sf::Lines);
        }
    }

    void drawStones() {
        float radius = kCellSize / 2.4f;
        for (int y = 0; y < kBoardSize; ++y) {
            for (int x = 0; x < kBoardSize; ++x) {
                Player player = board_.at(x, y);
                if (player == Player::None) {
                    continue;
                }
                sf::CircleShape stone(radius);
                stone.setOrigin(radius, radius);
                stone.setPosition(kPadding + x * kCellSize, kPadding + y * kCellSize);
                if (player == Player::Black) {
                    stone.setFillColor(sf::Color::Black);
                } else {
                    stone.setFillColor(sf::Color::White);
                    stone.setOutlineColor(sf::Color::Black);
                    stone.setOutlineThickness(2);
                }
                window_.draw(stone);
            }
        }
    }

    void drawHud() {
        if (!fontLoaded_) {
            return;
        }
        sf::Text text;
        text.setFont(font_);
        text.setCharacterSize(16);
        text.setFillColor(sf::Color::Black);
        std::string status = "R: reset";
        if (winner_.has_value()) {
            if (winner_.value() == Player::Black) {
                status = "Victoire noire ! R: reset";
            } else if (winner_.value() == Player::White) {
                status = "Victoire blanche ! R: reset";
            } else {
                status = "Match nul. R: reset";
            }
        }
        text.setString(status);
        text.setPosition(20.f, static_cast<float>(kWindowHeight - 60));
        window_.draw(text);

        sf::Text stats;
        stats.setFont(font_);
        stats.setCharacterSize(14);
        stats.setFillColor(sf::Color::Black);
        double nodesPerSecond = 0.0;
        if (lastAiTimeMs_ > 0) {
            nodesPerSecond = (ai_.nodes() * 1000.0) / lastAiTimeMs_;
        }
        stats.setString("Profondeur: " + std::to_string(ai_.lastCompletedDepth()) +
                        " | Score: " + std::to_string(ai_.lastScore()) +
                        " | Nodes/s: " + std::to_string(static_cast<int>(nodesPerSecond)));
        stats.setPosition(20.f, static_cast<float>(kWindowHeight - 36));
        window_.draw(stats);
    }

    sf::RenderWindow window_;
    GomokuBoard board_;
    GomokuAI ai_;
    Zobrist zobrist_;
    std::optional<Player> winner_;
    sf::Font font_;
    bool fontLoaded_;
    std::int64_t lastAiTimeMs_{};
};
} // namespace

int main() {
    GomokuGame game;
    game.run();
    return 0;
}

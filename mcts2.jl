using Base: @kwdef, isequal
using StaticArrays

abstract type DomState end

mutable struct State{D<:DomState}
    id::Int

    parent_id::Int  # state id, -1 for root state
    children::Vector{Int} # state id

    # UCB params
    n::Int
    t::Float32

    # DomState
    data::D

    function State(id, parent_id, data::D) where {D}
        s = new{D}()
        s.id = id
        s.parent_id = parent_id
        s.n = 0
        s.t = 0
        s.data = data
        return s
    end
end

struct Tree
    states::Vector{State{<:DomState}}
end

# accessors
curstate(tree) = tree.cur, tree.states[tree.cur]

function run_search(tree)
    iterations = 50
    for _ in 1:iterations
        s = select(tree)
        s = expand!(tree, s)
        v = rollout(s)
        backup!(tree, s, v)
    end
end

function select(tree::Tree)
    # start at root state
    s = first(tree.states)

    while !isleaf(s)
        # calculate children UCB and get the one with max
        children_ucb = zeros(size(s.children))
        for (si, sc) in enumerate(s.children)
            children_ucb[si] = ucb(tree, sc)
        end
        si = argmax(children_ucb)
        s = s.children[si]
    end

    return s
end

function expand!(tree::Tree, s::State)
    A = actions(s.data)
    for a in A
        apply!(tree, s, a)
    end

    child = rand(s.children)
    return child
end

function rollout(s_cur::State)
    dom_s = s_cur.data
    while !isterminal(dom_s)
        dom_s = apply(dom_s, rollout_one(dom_s))
    end
    v = terminal_value(dom_s)
    return v
end

function backup(tree, s, v) end

isleaf(s::State) = length(s.children) == 0
isterminal(s::State) = isterminal(s.data)
parent(tree::Tree, s::State)::State = tree.states[s.parent_id]

function isterminal(data)
    error("not implemented")
end

function terminal_value(s)
    error("not implemented")
end

# base ucb1 formula
ucb(v, parent_n, child_n) = v + 2 * sqrt(ln(parent_n) / child_n)

function ucb(tree, s)
    p = parent(tree, s)
    parent_n = visit_count(tree, p)
    child_n = visit_count(tree, s)
    v = total(tree, s) / n
    return ucb(v, parent_n, child_n)
end

function apply!(tree::Tree, s::State, a)
    data = apply(s.data, a)
    id = length(tree.states) + 1
    child = State{eltype(tree.states)}(id, s.id, data)
    push!(tree.states, child)
    push!(s.children, id)
    return child
end



### TTT
const Board = MMatrix{3,3,Int8}
const Move = Int8

@enum Player::Int8 PX=1 PO=-1

struct TicTacToe <: DomState
    player::Player
    board::Board
end

isequal(x::TicTacToe,y::TicTacToe) = isequal(x.player, y.player) && isequal(x.board, y.board)

flip(p::Player) = -Int(p)

function run_search(::Type{TicTacToe})
    board = zeros(Board)
    data = TicTacToe(PX, board)
    s0 = State(0, -1, data)
    tree = Tree{TicTacToe}([s0], [])
    return run_search(tree)
end

function actions(data::TicTacToe)
    return findall(x -> x == 0, data.board)
end

function apply(data::TicTacToe, a::Move)
    d = TicTacToe(flip(data.player),copy(data.board))
    x = a % 3
    y = a ÷ 3
    d.board[x,y] = Int(data.player)
    return d
end

function rollout_one(data::TicTacToe)
    A = actions(data)
    return rand(A)
end

function isterminal(data::TicTacToe)
    cval = abs.(sum(data.board, dims = 1)) .= 3
    rval = abs.(sum(data.board, dims = 2)) .= 3
    return sum(cval) > 0 || sum(rval) > 0
end

function terminal_value(data::TicTacToe)
end
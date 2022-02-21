
using Base: @kwdef

@kwdef struct Tree{S}
    states::Array{S} = []
    total_value::Array{Int32} = []
    visit_count::Array{Int32} = []
    children::Array{Array{Int32}} = []
end

struct Node{S}
    tree::Tree{S}
    state::S
end

struct RandomGameState
    # mandatory node fields
    id::Int32
    parent_id::Int32

    # domain fields
    action::Int32
end

const RandomNode = Node{RandomGameState}
const RandomTree = Tree{RandomGameState}


function simulate(tree::Tree)
    node = traverse_and_expand!(tree)
    rollout(node)
    backprop()
end


function traverse_and_expand!(tree)
    c_node = start_node!(tree)
    while !is_leaf(c_node)
        c_node = get_max_ucb_child(c_node)
    end

    n = get_visit_count(c_node)
    if n == 0
        return c_node
    else
        first_child = expand_node!(c_node)
        return first_child
    end
end

function expand_node!(node)
    actions = get_available_actions(node)
    child_nodes = get_child_nodes_from_actions!(node, actions)
    return first(child_nodes)
end

function calc_ucb(node)
    n = get_visit_count(node)
    if n == 0
        return Inf
    end
    t = get_total_value(node)
    N = get_parent_visit_count(node)
    return (t / n) + 2(sqrt(log(N) / log(n)))
end

function rollout(node)
    tree = node.tree
    c_node = node
    while true
        if is_terminal_node(c_node)
            return node_value(c_node)
        end
        action = get_random_action(c_node)
        c_node = Node(tree, state_from_action(c_node, action, 0))
    end
end

function start_node!(tree)
    if length(tree.states) == 0
        node = add_root_node!(tree)
        return node
    else
        return Node(tree, first(tree.states))
    end
end

next_state_id(tree) = length(tree.states) + 1

function add_root_node!(tree)
    state = create_state(tree, next_state_id(tree), -1)
    push_state!(tree, state)
    return Node(tree, state)
end


function push_state!(tree, state)
    push!(tree.states, state)
    push!(tree.total_value, 0)
    push!(tree.visit_count, 0)
    push!(tree.children, [])
end

function get_max_ucb_child(node)
    max_ucb = -Inf32
    max_child = nothing
    for child in get_children(node)
        ucb = calc_ucb(child)
        if ucb > max_ucb
            max_ucb = ucb
            max_child = child
        end
    end

    @assert !isnothing(max_child)
    return max_child
end

# node functions
get_total_value(node) = node.tree.total_value[node.state.id]
get_visit_count(node) = node.tree.visit_count[node.state.id]
get_children(node) = [node.tree.states[i] for i in node.tree.children[node.state.id]]
get_parent_visit_count(node) = node.tree.visit_count[node.state.parent_id]
is_leaf(node) = length(node.tree.children[node.state.id]) == 0


function get_child_nodes_from_actions!(node, actions)
    out = []
    for action in actions
        child_node = add_node_from_action!(node, action)
        push!(out, child_node)
    end
    return out
end

function add_node_from_action!(node, action)
    child_state = state_from_action(node, action, next_state_id(tree))
    push_state!(tree, child_state)
    return Node(tree, child_state)
end

# domain functions


function get_available_actions(node)
    n = rand(1:5)
    return rand(0:20, n)
end

create_state(tree::Tree{RandomGameState}, id, parent_id) =
    RandomGameState(id, parent_id, -1)

is_terminal_node(node::RandomNode) = rand() < 0.10  # 10% chance of being a terminal node
node_value(node::RandomNode) = node.state.action
state_from_action(node::RandomNode, action, child_id) =
    RandomGameState(child_id, node.state.parent_id, action)
get_random_action(node) = first(get_available_actions(node))
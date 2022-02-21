@kwdef mutable struct Node{A,S}
    state::S  # s
    to_play::Int8 = 1  # 1 = white, -1 = black
    visit_count::Int32 = 0  # N(s, a)
    value_sum::Float32 = 0  # W(s, a)
    prior::Float32 = 0
    children::Dict{A,S} = {}
end

# W / N
q_value(node) = node.visit_count > 0 ? node.value_sum / node.visit_count : 0

# U(s, a) = (C(s)*P(s, a) * sqrt(N(s)))/(1 + N(s,a))
# C(s) = ...


function search(config, game, network)
    root = root_node(game)
    for sim in config.num_simulations
        node = root
        search_path = [node]
        scratch_game = clone(game)
        while expanded(node)
            action, node = select_child(config, node)
            apply_action!(scratch_game, action)
            push!(search_path, node)
        end

        value = evaluate!(node, scratch_game, network)
        backpropagate(search_path, value, to_play(scratch_game))
    end
    return select_action(config, game, root), root
end

function select_child(config, node)
    _, action, child = max(
        (ucb_score(config, node, child), action, child) for
        (action, child) in node.children
    )
    return action, child
end

function simulate(tree)
    s = root(tree)
    while not
        is_leaf(tree, s)
        action = choose_action()
        push!((node, a), visited)
        s′ = execute_move(node, a)
    end
    expand_leaf(s′)
end


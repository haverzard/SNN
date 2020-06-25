struct GenerationError <: Exception
    type::String
    cause::String
end